
import os
import jax
import jax.numpy as jnp
from flax import nnx
from ponderttt.data import get_tokenizer
from ponderttt.models import load_ttt_model
from ponderttt.models.gating_nnx import GatingConfig, GatingNetwork
from ponderttt.experiments.run_benchmarks import SimpleGenerator, _policy_forward_jit, _gating_forward_jit
from ponderttt.utils.checkpointing import load_checkpoint

# Helper to unwrap state
def unwrap_state(state):
    if isinstance(state, dict):
        if "value" in state and len(state) == 1:
            return state["value"]
        return {k: unwrap_state(v) for k, v in state.items()}
    return state

class DebugGenerator(SimpleGenerator):
    """Generator that prints gating decisions."""
    
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        # Tokenize
        prompt_token_ids = self._truncate_context(self.tokenizer.encode(prompt).ids)
        input_ids = jnp.array([prompt_token_ids], dtype=jnp.int32)
        generated_tokens = []
        
        print(f"\nGenerating (Prompt len: {len(prompt_token_ids)})...")
        
        for i in range(max_new_tokens):
            current_len = input_ids.shape[1]
            
            # Alignment logic (same as SimpleGenerator)
            mini_batch_size = 16
            remat_group_size = 1
            if hasattr(self.model, "fast_layer") and hasattr(self.model.fast_layer, "config"):
                config = self.model.fast_layer.config
                if hasattr(config, 'mini_batch_size'):
                    mini_batch_size = int(getattr(config, 'mini_batch_size'))
                if hasattr(config, 'remat_mini_batch_group_size'):
                    remat_group_size = int(getattr(config, 'remat_mini_batch_group_size'))

            alignment = mini_batch_size * remat_group_size
            target_len = self._get_target_length(current_len, alignment)
            pad_len = target_len - current_len
            
            if pad_len > 0:
                pads = jnp.full((1, pad_len), self.pad_token_id, dtype=jnp.int32)
                padded_input = jnp.concatenate([input_ids, pads], axis=1)
                attention_mask = jnp.concatenate([
                    jnp.ones((1, current_len), dtype=jnp.int32),
                    jnp.zeros((1, pad_len), dtype=jnp.int32)
                ], axis=1)
            else:
                padded_input = input_ids
                attention_mask = jnp.ones((1, current_len), dtype=jnp.int32)
            
            gating_scale = None
            
            if self.gating_net is not None:
                out_base = self._call_model(padded_input, use_ttt=False, gating_scale=None)

                features = self.extract_features_jit(
                    input_ids=padded_input,
                    logits=out_base["logits"],
                    attention_mask=attention_mask
                )

                if hasattr(self.gating_net, "evaluate_actions"):
                    policy_out = _policy_forward_jit(self.gating_net, features, deterministic=True)
                    action = int(policy_out["action"][0])
                    step_map = [0, 1, 2, 4]
                    scale = float(step_map[action])
                    print(f"  Step {i+1}: RL Action={action} (Scale={scale})")
                else:
                    scale = float(_gating_forward_jit(self.gating_net, features, train=False)[0, 0])
                    # Print the gating scale decision
                    print(f"  Step {i+1}: Gating Scale = {scale:.4f}")

                gating_scale = jnp.array([[scale]])
                outputs = self._call_model(padded_input, use_ttt=True, gating_scale=gating_scale)
            else:
                print(f"  Step {i+1}: No Gating Network")
                outputs = self._call_model(padded_input, use_ttt=False, gating_scale=None)
            
            logits = outputs["logits"][:, current_len - 1, :]
            
            if temperature > 0:
                key = self._next_rng_key()
                scaled_logits = logits / float(temperature)
                next_token = jax.random.categorical(key, scaled_logits, axis=-1)
            else:
                next_token = jnp.argmax(logits, axis=-1)
            
            next_token = next_token.reshape(1, 1)
            token_val = int(next_token[0, 0])
            
            if token_val != self.eos_token_id:
                generated_tokens.append(token_val)
                print(f"    Generated token: {self.tokenizer.decode([token_val])!r}")
            else:
                print("    Generated EOS")

            input_ids = jnp.concatenate([input_ids, next_token], axis=1)
            if input_ids.shape[1] > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            if token_val == self.eos_token_id:
                break
                
        return self.tokenizer.decode(generated_tokens)

def main():
    model_scale = "125m"
    checkpoint_path = "outputs/diff/125m_budget1.5/checkpoint_10000"
    
    tokenizer = get_tokenizer("gpt2")
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    vocab_size = tokenizer.get_vocab_size()

    print(f"Loading base model {model_scale}...")
    model, _ = load_ttt_model(
        model_name="gpt2",
        fast_weight_type="ttt",
        load_pretrained=True,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id
    )
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    class TrainableSystem(nnx.Module):
        def __init__(self, ttt_model, gating_net):
            self.fast_layer = ttt_model.fast_layer
            self.fast_norm = ttt_model.fast_norm
            self.gating_net = gating_net
            if hasattr(ttt_model, 'lm_head'):
                self.lm_head = ttt_model.lm_head
            else:
                self.lm_head = None
    
    dummy_gating = GatingNetwork(GatingConfig(), nnx.Rngs(0))
    trainable_sys = TrainableSystem(model, dummy_gating)
    
    # Loose loading logic
    try:
        ckpt = load_checkpoint(checkpoint_path, target=None)
        if "state" in ckpt and "model" in ckpt["state"]:
            model_state = unwrap_state(ckpt["state"]["model"])
            nnx.update(trainable_sys, model_state)
            print("Loaded checkpoint successfully")
            gating_net = trainable_sys.gating_net
        else:
            raise ValueError("Checkpoint format invalid")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Use DebugGenerator
    generator = DebugGenerator(model, tokenizer, gating_net)
    
    # Test prompts (Simulate HumanEval structure)
    from ponderttt.evaluation.benchmarks import CodeProblem
    os.environ["PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS"] = "1"

    # Create a fake problem resembling a simple HumanEval task
    problem = CodeProblem(
        task_id="debug/0",
        prompt="def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
        canonical_solution="    return a + b",
        test_code="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        entry_point="add"
    )
    
    print(f"\nPrompt:\n{problem.prompt}")
    print("-" * 20)
    completion = generator.generate(problem.prompt, max_new_tokens=20, temperature=0.0)
    print(f"\nFinal Completion:\n{completion!r}")
    print("-" * 20)

    # Debug the execution check
    print("Debugging _check_solution execution...")
    
    source = f"{problem.prompt}\n{completion}\n"
    print(f"Full Source Code to Execute:\n'''\n{source}\n'''")
    
    try:
        # Manually run the check logic to catch specific errors
        namespace = {}
        exec(compile(source, "<completion>", "exec"), namespace, namespace)
        print(">>> Compilation & Definition Successful")
        
        exec(compile(problem.test_code, "<tests>", "exec"), namespace, namespace)
        print(">>> Test Code Compiled")
        
        if "check" in namespace and problem.entry_point in namespace:
            namespace["check"](namespace[problem.entry_point])
            print(">>> Check Function Passed!")
        else:
            print(f">>> Entry point '{problem.entry_point}' or 'check' not found in namespace.")
            print(f"    Keys found: {list(namespace.keys())}")
            
    except Exception as e:
        print(f">>> EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

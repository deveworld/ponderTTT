"""
Run code generation benchmarks using trained TTT models.

Usage:
    python -m ponderttt.experiments.run_benchmarks --benchmark humaneval --model_scale 125m --checkpoint outputs/my_ckpt
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from ..data import get_tokenizer
from ..evaluation.benchmarks import BenchmarkSuite, _check_solution
from ..models import TTTTransformerLM, load_ttt_model, TTTConfig
from ..models.gating_nnx import GatingConfig, GatingNetwork
from ..utils import FeatureExtractor, extract_features
from ..utils.checkpointing import load_checkpoint


# JIT compiled forward pass
# def _generate_step_impl(model, input_ids, use_ttt, gating_scale):
#     return model(input_ids, use_ttt=use_ttt, gating_scale=gating_scale)

# generate_step = cast(Callable, nnx.jit(_generate_step_impl, static_argnames=("use_ttt",)))


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument(
        "--benchmark", 
        type=str, 
        choices=["humaneval", "mbpp", "classeval", "all"], 
        default="humaneval"
    )
    parser.add_argument("--model_scale", type=str, default="125m")
    parser.add_argument("--checkpoint", type=str, help="Path to trained TTT checkpoint (optional)")
    parser.add_argument("--baseline", type=str, help="Path to baseline model checkpoint (optional)")
    parser.add_argument("--gating_checkpoint", type=str, help="Path to gating network checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems (for testing)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_unsafe", action="store_true", help="Allow unsafe code execution")
    return parser.parse_args()


class SimpleGenerator:
    """Simple autoregressive generator for NNX models."""
    
    def __init__(self, model: TTTTransformerLM, tokenizer, gating_net=None):
        self.model = model
        self.tokenizer = tokenizer
        self.gating_net = gating_net
        self.pad_token_id = tokenizer.token_to_id("<|pad|>")
        self.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        # GPT-2 variants cap usable context via learned position embeddings
        base_config = getattr(model, "gpt2_config", None)
        if base_config is None and hasattr(model, "base_model"):
            base_config = getattr(model.base_model, "config", None)
        self.max_seq_len = getattr(base_config, "n_positions", 1024)
        self._warned_truncation = False
        
        # Feature extractor for gating
        self.feature_extractor = FeatureExtractor(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=self.pad_token_id,
            seq_length_norm=512, # approximate norm
        )

        # JIT compiled functions
        @nnx.jit(static_argnames=("use_ttt",))
        def model_forward(model, input_ids, use_ttt, gating_scale):
            return model(input_ids, use_ttt=use_ttt, gating_scale=gating_scale)
        self.model_forward = model_forward

        @nnx.jit(static_argnames=("deterministic",))
        def policy_forward(net, x, deterministic):
            return net(x, deterministic=deterministic)
        self.policy_forward = policy_forward

        @nnx.jit(static_argnames=("train",))
        def gating_forward(net, x, train):
            return net(x, train=train)
        self.gating_forward = gating_forward
        
        # JIT compiled feature extraction
        # We use the standalone function to avoid state issues and allow JIT
        vocab_size = tokenizer.get_vocab_size()
        pad_id = self.pad_token_id
        
        @jax.jit
        def extract_features_jit(input_ids, logits, attention_mask):
            return extract_features(
                input_ids=input_ids,
                logits=logits,
                vocab_size=vocab_size,
                pad_token_id=pad_id,
                seq_length_norm=512.0,
                attention_mask=attention_mask,
                budget_remaining=1.0
            )
        self.extract_features_jit = extract_features_jit

    def _truncate_context(self, token_ids: list[int]) -> list[int]:
        """Clamp context to the model's positional window."""
        if len(token_ids) <= self.max_seq_len:
            return token_ids
        if not self._warned_truncation:
            print(
                f"WARNING: Prompt length {len(token_ids)} exceeds model limit {self.max_seq_len}. "
                "Keeping the most recent tokens."
            )
            self._warned_truncation = True
        return token_ids[-self.max_seq_len:]

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 128, 
        temperature: float = 0.0
    ) -> str:
        """Generate completion for a prompt."""
        # Tokenize
        prompt_token_ids = self._truncate_context(self.tokenizer.encode(prompt).ids)
        input_ids = jnp.array([prompt_token_ids], dtype=jnp.int32)  # [1, seq_len]
        generated_tokens: list[int] = []
        
        # Generation loop (inefficient, no KV cache)
        for _ in range(max_new_tokens):
            current_len = input_ids.shape[1]
            
            # Prepare TTT inputs (padding if needed)
            mini_batch_size = 16
            remat_group_size = 1
            if hasattr(self.model, "fast_layer") and hasattr(self.model.fast_layer, "config"):
                if isinstance(self.model.fast_layer.config, TTTConfig):
                    mini_batch_size = self.model.fast_layer.config.mini_batch_size
                    remat_group_size = self.model.fast_layer.config.remat_mini_batch_group_size
            
            alignment = mini_batch_size * remat_group_size
            pad_len = (alignment - (current_len % alignment)) % alignment
            
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
            
            # Determine gating / TTT usage
            use_ttt = False
            gating_scale = None
            
            # We need the baseline output anyway if we are gating OR if we end up skipping
            # To avoid double computation, we compute baseline first if gating is enabled
            
            if self.gating_net is not None:
                out_base = self.model_forward(self.model, padded_input, use_ttt=False, gating_scale=None)

                features = self.extract_features_jit(
                    input_ids=padded_input,
                    logits=out_base["logits"],
                    attention_mask=attention_mask
                )

                if hasattr(self.gating_net, "evaluate_actions"):
                    policy_out = self.policy_forward(self.gating_net, features, deterministic=True)
                    action = int(policy_out["action"][0])
                    step_map = [0, 1, 2, 4]
                    scale = float(step_map[action])
                else:
                    scale = float(self.gating_forward(self.gating_net, features, train=False)[0, 0])

                if scale > 0.01:
                    use_ttt = True
                    gating_scale = jnp.array([[scale]])
                    outputs = self.model_forward(self.model, padded_input, use_ttt=True, gating_scale=gating_scale)
                else:
                    outputs = out_base
            else:
                outputs = self.model_forward(self.model, padded_input, use_ttt=False, gating_scale=None)
            
            # Get logits for the last REAL token
            logits = outputs["logits"][:, current_len - 1, :]  # [1, vocab_size]
            
            # Sampling
            if temperature > 0:
                probs = jax.nn.softmax(logits / temperature, axis=-1)
                next_token = jax.random.categorical(jax.random.PRNGKey(0), jnp.log(probs)) 
            else:
                next_token = jnp.argmax(logits, axis=-1)
            
            next_token = next_token.reshape(1, 1)
            token_val = int(next_token[0, 0])
            if token_val != self.eos_token_id:
                generated_tokens.append(token_val)
            input_ids = jnp.concatenate([input_ids, next_token], axis=1)
            if input_ids.shape[1] > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            if token_val == self.eos_token_id:
                break
                
        # Decode
        completion = self.tokenizer.decode(generated_tokens)
        return completion

    def generate_batch(
        self, 
        prompts: list[str], 
        max_new_tokens: int = 128, 
        temperature: float = 0.0
    ) -> list[str]:
        """Generate completions for a batch of prompts."""
        # Tokenize
        input_ids_list = [self._truncate_context(list(self.tokenizer.encode(p).ids)) for p in prompts]
        batch_size = len(prompts)
        generated_tokens = [[] for _ in range(batch_size)]
        
        # Track which sequences are finished
        finished = [False] * batch_size
        
        # JIT compilation requires static shapes, but we have dynamic loop.
        # We can't easily JIT the whole loop with dynamic batching/padding.
        # But the inner model call is JITted.
        
        for _ in range(max_new_tokens):
            if all(finished):
                break
                
            # Prepare batch
            for idx in range(batch_size):
                input_ids_list[idx] = self._truncate_context(input_ids_list[idx])

            current_lens = [len(ids) for ids in input_ids_list]
            max_len = max(current_lens)
            
            # Alignment
            mini_batch_size = 16
            remat_group_size = 1
            if hasattr(self.model, "fast_layer") and hasattr(self.model.fast_layer, "config"):
                if isinstance(self.model.fast_layer.config, TTTConfig):
                    mini_batch_size = self.model.fast_layer.config.mini_batch_size
                    remat_group_size = self.model.fast_layer.config.remat_mini_batch_group_size
            
            alignment = mini_batch_size * remat_group_size
            target_len = ((max_len + alignment - 1) // alignment) * alignment
            target_len = min(target_len, self.max_seq_len)
            
            # Construct tensors
            padded_input_ids = []
            attention_masks = []
            
            for i, ids in enumerate(input_ids_list):
                pad_len = target_len - len(ids)
                # Right padding for input_ids to keep prefix contiguous?
                # No, if we right pad, we must ensure the model attends correctly.
                # GPT-2 uses absolute position embeddings.
                # If we right pad: [A, B, C, P, P]
                # Pos ids: 0, 1, 2, 3, 4
                # Real tokens are at 0, 1, 2.
                # Next token should be at 3.
                # This works fine with standard causal attention.
                padded_ids = ids + [self.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
                padded_input_ids.append(padded_ids)
                attention_masks.append(mask)
                
            input_tensor = jnp.array(padded_input_ids, dtype=jnp.int32)
            mask_tensor = jnp.array(attention_masks, dtype=jnp.int32)
            
            # Gating / TTT
            use_ttt = False
            gating_scale = None
            
            if self.gating_net is not None:
                out_base = self.model_forward(self.model, input_tensor, use_ttt=False, gating_scale=None)

                features = self.extract_features_jit(
                    input_ids=input_tensor,
                    logits=out_base["logits"],
                    attention_mask=mask_tensor
                )

                if hasattr(self.gating_net, "evaluate_actions"):
                    policy_out = self.policy_forward(self.gating_net, features, deterministic=True)
                    actions = policy_out["action"]
                    step_map = jnp.array([0.0, 1.0, 2.0, 4.0])
                    scales = step_map[actions.astype(int)]
                else:
                    scales = self.gating_forward(self.gating_net, features, train=False)[:, 0]

                if jnp.any(scales > 0.01):
                    use_ttt = True
                    gating_scale = scales[:, None]
                    outputs = self.model_forward(self.model, input_tensor, use_ttt=True, gating_scale=gating_scale)
                else:
                    outputs = out_base
            else:
                outputs = self.model_forward(self.model, input_tensor, use_ttt=False, gating_scale=None)
            
            # Extract logits
            indices = jnp.array([l - 1 for l in current_lens]) # [B]
            next_token_logits = outputs["logits"][jnp.arange(batch_size), indices, :] # [B, V]
            
            # Sampling
            if temperature > 0:
                probs = jax.nn.softmax(next_token_logits / temperature, axis=-1)
                next_tokens = jax.random.categorical(jax.random.PRNGKey(0), jnp.log(probs)) 
            else:
                next_tokens = jnp.argmax(next_token_logits, axis=-1)
            
            # Update sequences
            for i, token in enumerate(next_tokens):
                if not finished[i]:
                    token_val = int(token)
                    if token_val != self.eos_token_id:
                        generated_tokens[i].append(token_val)
                    if token_val == self.eos_token_id:
                        finished[i] = True
                    else:
                        input_ids_list[i].append(token_val)
                        if len(input_ids_list[i]) > self.max_seq_len:
                            # Keep only the most recent tokens within context window
                            input_ids_list[i] = input_ids_list[i][-self.max_seq_len:]
                        
        # Decode
        completions = []
        for i in range(batch_size):
            completions.append(self.tokenizer.decode(generated_tokens[i]))
            
        return completions


def unwrap_state(state):
    """Unwrap NNX state dictionary if it contains 'value' keys."""
    if isinstance(state, dict):
        # Check if this dict represents a Variable (has 'value' key)
        # Note: NNX variables serialized by Orbax might appear as {'value': array}
        if "value" in state and len(state) == 1:
            return state["value"]
        return {k: unwrap_state(v) for k, v in state.items()}
    return state


def main():
    args = parse_args()

    if args.checkpoint and args.baseline:
        raise ValueError("Please provide only one of --checkpoint or --baseline")

    if args.allow_unsafe:
        os.environ["PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS"] = "1"
        print("WARNING: Unsafe code execution enabled!")
    elif os.environ.get("PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS") != "1":
        print("\nERROR: Benchmarks require executing generated code.")
        print("Please use --allow_unsafe to enable code execution.")
        print("Example: python -m ponderttt.experiments.run_benchmarks --allow_unsafe ...\n")
        return
    
    tokenizer = get_tokenizer({"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale])
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    vocab_size = tokenizer.get_vocab_size()

    # Load Model
    print(f"Loading model {args.model_scale}...")
    model, _ = load_ttt_model(
        model_name={"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale],
        fast_weight_type="ttt",
        load_pretrained=True,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id
    )
    
    gating_net = None

    # Load Checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        
        # Helper to reconstruct TrainableSystem for loading diff checkpoints
        class TrainableSystem(nnx.Module):
            def __init__(self, ttt_model, gating_net):
                self.fast_layer = ttt_model.fast_layer
                self.fast_norm = ttt_model.fast_norm
                self.gating_net = gating_net
                if hasattr(ttt_model, 'lm_head'):
                    self.lm_head = ttt_model.lm_head
                else:
                    self.lm_head = None
        
        # Create a dummy gating net to match structure if needed
        dummy_gating = GatingNetwork(GatingConfig(), nnx.Rngs(0))
        trainable_sys = TrainableSystem(model, dummy_gating)
        
        try:
            # Try loading as Differentiable Training checkpoint
            # Use target to ensure correct structure restoration
            print("Attempting to load with strict structure matching...")
            target = {"state": {"model": nnx.state(trainable_sys)}}
            
            ckpt = load_checkpoint(args.checkpoint, target=target)
            
            if "state" in ckpt and "model" in ckpt["state"]:
                nnx.update(trainable_sys, ckpt["state"]["model"])
                print("Loaded as Differentiable Training checkpoint")
            else:
                raise ValueError("Checkpoint does not contain 'state.model'")

            # If we successfully loaded, we might want to use the loaded gating net if not provided separately
            if args.gating_checkpoint is None:
                 print("Using gating network from model checkpoint")
                 gating_net = trainable_sys.gating_net

        except Exception as e_diff:
            print(f"Strict loading failed ({e_diff}), trying loose loading...")
            try:
                # Fallback: Load without target
                ckpt = load_checkpoint(args.checkpoint, target=None)
                
                if "state" in ckpt and "model" in ckpt["state"]:
                    # Try to update trainable_sys with loose dict
                    # Unwrap 'value' keys if present (fix for KeyError: Ellipsis)
                    model_state = unwrap_state(ckpt["state"]["model"])
                    nnx.update(trainable_sys, model_state)
                    print("Loaded as Differentiable Training checkpoint (loose)")
                    if args.gating_checkpoint is None:
                        gating_net = trainable_sys.gating_net
                else:
                    raise ValueError("Checkpoint does not contain 'state.model'")
            except Exception as e_loose:
                print(f"Not a Differentiable Training checkpoint ({e_loose}), trying Baseline...")
                try:
                    # Try loading as Baseline checkpoint (Model structure)
                    # Load without target
                    ckpt = load_checkpoint(args.checkpoint, target=None)
                    
                    if "state" in ckpt and "model" in ckpt["state"]:
                        nnx.update(model, ckpt["state"]["model"])
                        print("Loaded as Baseline checkpoint")
                    else:
                        raise ValueError("Checkpoint does not contain 'state.model'")
                except Exception as e_base:
                     print(f"Failed to load checkpoint: {e_base}")
                     raise ValueError("Could not load checkpoint as either Differentiable or Baseline format")
    elif args.baseline:
        print(f"Loading baseline checkpoint from {args.baseline}...")
        ckpt = load_checkpoint(args.baseline, target=None)
        if "state" in ckpt and "model" in ckpt["state"]:
            nnx.update(model, ckpt["state"]["model"])
            print("Loaded baseline checkpoint")
        else:
            raise ValueError("Baseline checkpoint does not contain 'state.model'")

    # Load Gating/Policy Checkpoint if provided
    if args.gating_checkpoint:
        print(f"Loading gating checkpoint from {args.gating_checkpoint}...")
        # Try PolicyNetwork first
        try:
            from ..models import PolicyConfig, PolicyNetwork
            # Assume default config or infer? hard to infer. using default for now.
            p_config = PolicyConfig(feature_dim=32, hidden_dim=128, num_actions=4)
            p_net = PolicyNetwork(p_config, nnx.Rngs(0))
            # Load without target
            ckpt = load_checkpoint(args.gating_checkpoint, target=None)
            if "state" in ckpt and "policy" in ckpt["state"]:
                nnx.update(p_net, ckpt["state"]["policy"])
                gating_net = p_net
                print("Loaded PolicyNetwork")
            else:
                raise ValueError("Checkpoint does not contain 'state.policy'")
        except Exception as e_policy:
             # Try GatingNetwork (part of differentiable training checkpoint)
             try:
                 g_config = GatingConfig(feature_dim=32, hidden_dim=64, scale_output=4.0)
                 g_net = GatingNetwork(g_config, nnx.Rngs(0))
                 # GatingNetwork is part of the differentiable model, so use model format
                 # Load without target
                 ckpt = load_checkpoint(args.gating_checkpoint, target=None)
                 if "state" in ckpt and "model" in ckpt["state"]:
                     nnx.update(g_net, ckpt["state"]["model"])
                     gating_net = g_net
                     print("Loaded GatingNetwork")
                 else:
                     raise ValueError("Checkpoint does not contain 'state.model'")
             except Exception as e:
                 print(f"Failed to load gating checkpoint as PolicyNetwork ({e_policy}) or GatingNetwork ({e})")
                 raise
    
    # Initialize Generator
    generator = SimpleGenerator(model, tokenizer, gating_net)
    
    # Select Benchmarks
    suite = BenchmarkSuite(
        include_humaneval=(args.benchmark in ["humaneval", "all"]),
        include_mbpp=(args.benchmark in ["mbpp", "all"]),
        include_classeval=(args.benchmark in ["classeval", "all"]),
    )
    
    results = {}
    
    # Run Evaluation
    for name, benchmark in suite.benchmarks.items():
        print(f"\nRunning {name} ({len(benchmark)} problems)...")
        
        problems = benchmark.problems
        if args.limit:
            problems = problems[:args.limit]
            print(f"  Limiting to {args.limit} problems")
            
        print(f"  Processing in batches of {args.batch_size}...")
        
        scores = []
        attempts_per_problem = []
        
        # Batch processing loop
        for i in tqdm(range(0, len(problems), args.batch_size), desc=f"Running {name}"):
            batch_problems = problems[i : i + args.batch_size]
            prompts = [p.prompt for p in batch_problems]
            
            try:
                completions = generator.generate_batch(prompts, args.max_new_tokens, args.temperature)
                
                for problem, completion in zip(batch_problems, completions):
                    # Check correctness (k=1)
                    is_correct = _check_solution(problem, completion)
                    scores.append(1.0 if is_correct else 0.0)
                    attempts_per_problem.append(1)
                    
            except Exception as e:
                print(f"  Batch failed: {e}")
                traceback.print_exc()
                # Fill with failures
                for _ in batch_problems:
                    scores.append(0.0)
                    attempts_per_problem.append(0)
            
            # Progress update
            # current_avg = sum(scores) / len(scores) if scores else 0.0
            # print(f"  Batch {i//args.batch_size + 1}/{(len(problems)+args.batch_size-1)//args.batch_size} done. Current Pass@1: {current_avg:.2%}")

        # Final results
        final_score = sum(scores) / len(scores) if scores else 0.0
        results[name] = {"pass@1": final_score}
        print(f"  Result: {final_score}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")

if __name__ == "__main__":
    main()

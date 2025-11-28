
from flax import nnx
from ponderttt.data import get_tokenizer
from ponderttt.models import load_ttt_model
from ponderttt.models.gating_nnx import GatingConfig, GatingNetwork
from ponderttt.experiments.run_benchmarks import SimpleGenerator
from ponderttt.utils.checkpointing import load_checkpoint

# Helper from run_benchmarks.py
def unwrap_state(state):
    """Unwrap NNX state dictionary if it contains 'value' keys."""
    if isinstance(state, dict):
        if "value" in state and len(state) == 1:
            return state["value"]
        return {k: unwrap_state(v) for k, v in state.items()}
    return state

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
    
    # Create a dummy gating net to match structure
    dummy_gating = GatingNetwork(GatingConfig(), nnx.Rngs(0))
    trainable_sys = TrainableSystem(model, dummy_gating)
    gating_net = None

    try:
        # Try loading as Differentiable Training checkpoint
        print("Attempting to load with strict structure matching...")
        target = {"state": {"model": nnx.state(trainable_sys)}}
        
        ckpt = load_checkpoint(checkpoint_path, target=target)
        
        if "state" in ckpt and "model" in ckpt["state"]:
            nnx.update(trainable_sys, ckpt["state"]["model"])
            print("Loaded as Differentiable Training checkpoint")
            gating_net = trainable_sys.gating_net
        else:
            raise ValueError("Checkpoint does not contain 'state.model'")

    except Exception as e_diff:
        print(f"Strict loading failed ({e_diff}), trying loose loading...")
        try:
            # Fallback: Load without target
            ckpt = load_checkpoint(checkpoint_path, target=None)
            
            if "state" in ckpt and "model" in ckpt["state"]:
                model_state = unwrap_state(ckpt["state"]["model"])
                nnx.update(trainable_sys, model_state)
                print("Loaded as Differentiable Training checkpoint (loose)")
                gating_net = trainable_sys.gating_net
            else:
                raise ValueError("Checkpoint does not contain 'state.model'")
        except Exception as e_loose:
            print(f"Failed to load checkpoint: {e_loose}")
            return

    generator = SimpleGenerator(model, tokenizer, gating_net)
    
    prompt = "def add(a, b):\n    'Return the sum of a and b.'\n"
    print(f"Prompt:\n{prompt}")
    print("-" * 20)
    
    # Generate with TTT enabled (SimpleGenerator handles this if gating_net is present)
    completion = generator.generate(prompt, max_new_tokens=50, temperature=0.0)
    print(f"Completion:\n{completion}")
    print("-" * 20)
    
    # Try a HumanEval style prompt
    prompt_he = "def is_palindrome(string: str) -> bool:\n    'Test if given string is a palindrome'\n    return"
    print(f"Prompt HE:\n{prompt_he}")
    print("-" * 20)
    completion_he = generator.generate(prompt_he, max_new_tokens=50, temperature=0.0)
    print(f"Completion HE:\n{completion_he}")

if __name__ == "__main__":
    main()
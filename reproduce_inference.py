
import jax
import jax.numpy as jnp
from flax import nnx
from ponderttt.models.gating_nnx import BinaryGatingConfig, BinaryGatingNetwork
from ponderttt.utils.checkpointing import load_checkpoint

def unwrap_state(state):
    if isinstance(state, dict):
        if "value" in state and len(state) == 1:
            return state["value"]
        return {str(k) if isinstance(k, int) else k: unwrap_state(v) for k, v in state.items()}
    return state

def main():
    checkpoint_path = "outputs/hard_skip/125m_update0.5"
    print(f"Testing checkpoint: {checkpoint_path}")

    # 1. Initialize Network
    config = BinaryGatingConfig(feature_dim=32, hidden_dim=64, scale_when_update=1.0)
    rngs = nnx.Rngs(0)
    net = BinaryGatingNetwork(config, rngs)

    # 2. Load Checkpoint
    # We need to reconstruct the full structure to load the subset
    # In training, the structure was:
    # state = {"model": {"gating_net": ..., "fast_layer": ..., ...}, ...}
    
    try:
        ckpt = load_checkpoint(checkpoint_path, target=None)
        if "state" in ckpt and "model" in ckpt["state"]:
            full_model_state = unwrap_state(ckpt["state"]["model"])
            
            if "gating_net" in full_model_state:
                gating_state = full_model_state["gating_net"]
                print("Found 'gating_net' in checkpoint.")
                print("Keys:", gating_state.keys())
                
                # Update network
                nnx.update(net, gating_state)
                print("Weights updated successfully.")
                
                # Inspect bias
                print("Head Bias:", net.head.bias.value)
            else:
                print("ERROR: 'gating_net' not found in model state.")
                print("Available keys:", full_model_state.keys())
        else:
            print("ERROR: Invalid checkpoint structure.")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return

    # 3. Test Inference with Dummy Data
    # Random features with EMA injection
    print("\n--- Inference Test with EMA Injection ---")
    
    batch_size = 5
    
    # Simulate features with scale matching FeatureExtractor outputs
    dummy_features = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 32))
    
    # Inject History Features (Indices 22-25)
    # 22: difficulty (EMA of loss) -> Set to ~2.0 (Mean CE from training logs)
    dummy_features = dummy_features.at[:, 22].set(2.0) 
    # 23: difficulty_std -> Set to ~0.5
    dummy_features = dummy_features.at[:, 23].set(0.5)
    # 24: cost_ema -> Set to ~2.0 (Mix of 1.0 and 3.0)
    dummy_features = dummy_features.at[:, 24].set(2.0)
    # 25: budget_rem (0.0 to 1.0) -> Set to 0.5
    dummy_features = dummy_features.at[:, 25].set(0.5)
    
    _, _, probs_ema, _ = net(dummy_features, train=False)
    print("Update Probs (with EMA ~2.0):", probs_ema[:, 1])

    print("\n--- Inference Test (Zero EMA) ---")
    # Simulate reset state (EMA = 0.0)
    features_zero_ema = dummy_features
    features_zero_ema = features_zero_ema.at[:, 22].set(0.0)
    features_zero_ema = features_zero_ema.at[:, 23].set(0.0)
    features_zero_ema = features_zero_ema.at[:, 24].set(0.0)
    
    _, _, probs_zero_ema, _ = net(features_zero_ema, train=False)
    print("Update Probs (with EMA = 0.0):", probs_zero_ema[:, 1])


if __name__ == "__main__":
    main()

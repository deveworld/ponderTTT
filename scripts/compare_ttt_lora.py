"""
Compare TTT Layer vs LoRA performance and efficiency.

Usage:
    python scripts/compare_ttt_lora.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
from flax import nnx

from ponderttt.models import load_ttt_model, count_trainable_parameters


def compare_models():
    """Compare TTT and LoRA models."""
    print("=" * 70)
    print("TTT Layer vs LoRA Comparison")
    print("=" * 70)

    # Test configuration
    batch_size = 4
    seq_len = 512
    model_name = "gpt2"

    print(f"\nTest setup:")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print()

    # 1. Create TTT model
    print("1. Loading TTT model...")
    ttt_model, _ = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        load_pretrained=False,
    )
    ttt_trainable, ttt_total = count_trainable_parameters(ttt_model)

    print(f"   ✓ TTT model loaded")
    print(f"     - Total params: {ttt_total:,}")
    print(f"     - Trainable: {ttt_trainable:,} ({ttt_trainable/ttt_total*100:.1f}%)")

    # 2. Create LoRA model (different ranks)
    results = []
    for rank in [64, 128, 256]:
        print(f"\n2. Loading LoRA model (rank={rank})...")

        from ponderttt.models import LoRAConfig

        lora_config = LoRAConfig(hidden_dim=768, rank=rank)
        lora_model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="lora",
            lora_config=lora_config,
            load_pretrained=False,
        )
        lora_trainable, lora_total = count_trainable_parameters(lora_model)

        print(f"   ✓ LoRA model loaded")
        print(f"     - Total params: {lora_total:,}")
        print(f"     - Trainable: {lora_trainable:,} ({lora_trainable/lora_total*100:.1f}%)")

        results.append(
            {
                "name": f"LoRA (r={rank})",
                "trainable": lora_trainable,
                "total": lora_total,
            }
        )

    # 3. Summary comparison
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(f"{'Method':<20} {'Trainable Params':>18} {'vs TTT':>12} {'Total':>15}")
    print("-" * 70)

    # TTT baseline
    print(
        f"{'TTT Layer':<20} {ttt_trainable:>18,} {'baseline':>12} {ttt_total:>15,}"
    )

    # LoRA variants
    for result in results:
        reduction = (1 - result["trainable"] / ttt_trainable) * 100
        print(
            f"{result['name']:<20} {result['trainable']:>18,} {reduction:>11.1f}% {result['total']:>15,}"
        )

    # 4. Recommendations
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)
    print("""
✅ Use TTT Layer when:
   - Maximum performance is critical
   - Academic reproducibility needed
   - GPU memory is sufficient
   - Following official TTT-LM implementation

✅ Use LoRA when:
   - Memory is limited (22-50% reduction)
   - Faster training needed
   - Experimenting with different ranks
   - Practical deployment scenarios

💡 Ablation study:
   - Main results: TTT Layer (academic justification)
   - Comparison: LoRA r=64/128 (practical alternative)
   - Paper contribution: Trade-off analysis
""")


if __name__ == "__main__":
    compare_models()

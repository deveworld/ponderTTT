"""
Quick demo experiment for validating the full pipeline.

This uses a tiny model and small dataset for fast iteration.
"""

import json
import os

import torch
from tqdm import tqdm

from ponderttt.data import get_wikitext_dataloaders
from ponderttt.models import TransformerConfig, TransformerTTT


def quick_demo():
    """Run a quick demo with tiny model."""
    print("=" * 60)
    print("PonderTTT Quick Demo")
    print("=" * 60)

    device = torch.device("cpu")

    # Load tiny dataset
    print("\n1. Loading WikiText-2...")
    train_loader, val_loader, test_loader = get_wikitext_dataloaders(
        dataset_name="wikitext-2-raw-v1",
        tokenizer_name="gpt2",
        max_length=64,  # Very short sequences
        batch_size=4,
        num_workers=0,
    )

    results = []

    # Test 1: Baseline Fixed-2
    print("\n2. Testing Fixed-2 baseline...")
    config = TransformerConfig(
        vocab_size=50257,
        hidden_dim=128,  # Very small
        num_layers=2,  # Only 2 layers
        num_heads=4,
        ffn_dim=512,
        max_seq_len=64,
        dropout=0.1,
        use_ttt=True,
        ttt_layer_idx=0,
        ttt_dim=64,
        ttt_iterations=2,
        use_adaptive_ttt=False,
    )

    model = TransformerTTT(config).to(device)
    print(f"   Model size: {sum(p.numel() for p in model.parameters()):,} params")

    # Quick eval (no training)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="   Evaluating", total=10)):
            if i >= 10:  # Only 10 batches
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"   Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

    results.append(
        {
            "config": "baseline",
            "ttt_iterations": 2,
            "test_loss": avg_loss,
            "test_perplexity": perplexity,
        }
    )

    # Test 2: Adaptive TTT
    print("\n3. Testing Adaptive TTT...")
    config = TransformerConfig(
        vocab_size=50257,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        max_seq_len=64,
        dropout=0.1,
        use_ttt=True,
        ttt_layer_idx=0,
        ttt_dim=64,
        ttt_iterations=2,
        use_adaptive_ttt=True,
        ttt_difficulty_metric="entropy",
        ttt_buckets=[1, 2, 4],
        ttt_target_distribution=[0.3, 0.4, 0.3],
    )

    model = TransformerTTT(config).to(device)

    # Quick eval
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_stats = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="   Evaluating", total=10)):
            if i >= 10:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels, return_stats=True)
            loss = outputs["loss"]

            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

            if "ttt_stats" in outputs:
                all_stats.extend(outputs["ttt_stats"])

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # Aggregate stats
    if all_stats:
        import numpy as np

        avg_iterations = np.mean([s["efficiency"]["avg_iterations"] for s in all_stats])
        flops_reduction = np.mean([s["efficiency"]["flops_reduction"] for s in all_stats])

        # Get allocation distribution
        distributions = [s["distribution"] for s in all_stats]
        avg_dist = {}
        for key in distributions[0].keys():
            avg_dist[key] = np.mean([d[key] for d in distributions])

        print(f"   Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        print(f"   Avg iterations: {avg_iterations:.2f}")
        print(f"   FLOPs reduction: {flops_reduction * 100:.1f}%")
        print(f"   Allocation: {avg_dist}")

        results.append(
            {
                "config": "adaptive",
                "test_loss": avg_loss,
                "test_perplexity": perplexity,
                "ttt_stats": {
                    "avg_iterations": avg_iterations,
                    "flops_reduction": flops_reduction,
                    "allocation_distribution": avg_dist,
                },
            }
        )

    # Print summary
    print("\n" + "=" * 60)
    print("DEMO RESULTS SUMMARY")
    print("=" * 60)

    for result in results:
        config = result["config"]
        ppl = result["test_perplexity"]

        print(f"\n{config.upper()}:")
        print(f"  Perplexity: {ppl:.2f}")

        if "ttt_stats" in result:
            stats = result["ttt_stats"]
            print(f"  Avg iterations: {stats['avg_iterations']:.2f}")
            print(f"  FLOPs reduction: {stats['flops_reduction'] * 100:.1f}%")
            print(f"  Allocation: {stats['allocation_distribution']}")

    print("\n" + "=" * 60)
    print("Demo complete! ✅")
    print("=" * 60)

    # Save results
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/quick_demo.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: experiments/results/quick_demo.json")


if __name__ == "__main__":
    quick_demo()

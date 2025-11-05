"""
Mini training experiment optimized for CPU.

Uses a small model and limited data for fast iteration and validation.
"""

import json
import math
import os

import torch
from tqdm import tqdm

from ponderttt.data import get_wikitext_dataloaders
from ponderttt.models import TransformerConfig, TransformerTTT


def train_mini_model(config_name, ttt_iterations, use_adaptive=False, num_epochs=2):
    """Train a tiny model quickly."""
    print(f"\n{'=' * 60}")
    print(f"Training: {config_name}")
    print(f"{'=' * 60}")

    device = torch.device("cpu")

    # Small dataset
    train_loader, val_loader, test_loader = get_wikitext_dataloaders(
        dataset_name="wikitext-2-raw-v1",
        tokenizer_name="gpt2",
        max_length=64,
        batch_size=8,
        num_workers=0,
    )

    # Tiny model
    config = TransformerConfig(
        vocab_size=50257,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
        max_seq_len=64,
        dropout=0.1,
        use_ttt=True,
        ttt_layer_idx=0,
        ttt_dim=64,
        ttt_iterations=ttt_iterations,
        use_adaptive_ttt=use_adaptive,
        ttt_difficulty_metric="entropy",
        ttt_buckets=[1, 2, 4],
        ttt_target_distribution=[0.3, 0.4, 0.3],
    )

    model = TransformerTTT(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    val_perplexities = []

    # Training
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        model.train()
        train_loss = 0.0
        train_tokens = 0

        pbar = tqdm(train_loader, desc="Training", total=100)
        for i, batch in enumerate(pbar):
            if i >= 100:  # Only 100 batches per epoch
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * input_ids.numel()
            train_tokens += input_ids.numel()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / train_tokens
        print(f"Train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Validation", total=50)):
                if i >= 50:
                    break

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]

                val_loss += loss.item() * input_ids.numel()
                val_tokens += input_ids.numel()

        avg_val_loss = val_loss / val_tokens
        val_ppl = math.exp(avg_val_loss)
        val_perplexities.append(val_ppl)
        print(f"Val perplexity: {val_ppl:.2f}")

    # Final test evaluation
    print("\nFinal test evaluation...")
    model.eval()
    test_loss = 0.0
    test_tokens = 0
    all_stats = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing", total=50)):
            if i >= 50:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels, return_stats=True)
            loss = outputs["loss"]

            test_loss += loss.item() * input_ids.numel()
            test_tokens += input_ids.numel()

            if "ttt_stats" in outputs and outputs["ttt_stats"]:
                all_stats.extend(outputs["ttt_stats"])

    test_ppl = math.exp(test_loss / test_tokens)
    print(f"Test perplexity: {test_ppl:.2f}")

    # Prepare results
    result = {
        "config": "baseline" if not use_adaptive else "adaptive",
        "ttt_iterations": ttt_iterations if not use_adaptive else "adaptive",
        "num_params": sum(p.numel() for p in model.parameters()),
        "val_perplexities": val_perplexities,
        "test_perplexity": test_ppl,
        "test_loss": test_loss / test_tokens,
    }

    # Add TTT stats for adaptive
    if use_adaptive and all_stats:
        import numpy as np

        avg_iterations = np.mean([s["efficiency"]["avg_iterations"] for s in all_stats])
        flops_reduction = np.mean([s["efficiency"]["flops_reduction"] for s in all_stats])

        distributions = [s["distribution"] for s in all_stats]
        avg_dist = {}
        for key in distributions[0].keys():
            avg_dist[key] = float(np.mean([d[key] for d in distributions]))

        result["ttt_stats"] = {
            "avg_iterations": float(avg_iterations),
            "flops_reduction": float(flops_reduction),
            "allocation_distribution": avg_dist,
        }

        print(f"Avg iterations: {avg_iterations:.2f}")
        print(f"FLOPs reduction: {flops_reduction * 100:.1f}%")

    return result


def main():
    """Run mini experiments."""
    print("=" * 60)
    print("PonderTTT Mini Training Experiment")
    print("Optimized for CPU - Quick validation")
    print("=" * 60)

    results = []

    # Run experiments
    configs = [
        ("Fixed-1", 1, False),
        ("Fixed-2", 2, False),
        ("Fixed-4", 4, False),
        ("Adaptive", 2, True),
    ]

    for config_name, ttt_iters, use_adaptive in configs:
        result = train_mini_model(config_name, ttt_iters, use_adaptive, num_epochs=2)
        results.append(result)

        # Save individual result
        os.makedirs("experiments/results", exist_ok=True)
        filename = f"mini_{result['config']}_{result['ttt_iterations']}.json"
        with open(f"experiments/results/{filename}", "w") as f:
            json.dump(result, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\n{'Config':<15} {'Test PPL':<12} {'Avg Iters':<12} {'FLOPs ↓':<10}")
    print("-" * 60)

    for result in results:
        config = result["config"]
        iters = result.get("ttt_iterations", "adaptive")
        ppl = result["test_perplexity"]

        config_str = f"{config}-{iters}"

        if "ttt_stats" in result:
            stats = result["ttt_stats"]
            avg_iters = stats["avg_iterations"]
            flops_red = stats["flops_reduction"]
            print(f"{config_str:<15} {ppl:<12.2f} {avg_iters:<12.2f} {flops_red * 100:<10.1f}%")
        else:
            print(f"{config_str:<15} {ppl:<12.2f} {iters:<12} {'-':<10}")

    print("\n" + "=" * 60)
    print("Mini experiment complete! ✅")
    print("\nRun analysis: uv run python experiments/analyze_wikitext2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

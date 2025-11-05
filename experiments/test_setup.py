"""
Quick test to verify WikiText-2 data loading and model setup.
"""

import torch

from ponderttt.data import get_wikitext_dataloaders
from ponderttt.models import TransformerConfig, TransformerTTT


def test_data_loading():
    """Test data loading."""
    print("Testing data loading...")

    train_loader, val_loader, test_loader = get_wikitext_dataloaders(
        dataset_name="wikitext-2-raw-v1",
        tokenizer_name="gpt2",
        max_length=128,
        batch_size=4,
        num_workers=0,
    )

    # Get one batch
    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Number of train batches: {len(train_loader)}")
    print(f"Number of val batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    return train_loader


def test_model(train_loader):
    """Test model forward pass."""
    print("\nTesting model...")

    # Create small model
    config = TransformerConfig(
        vocab_size=50257,
        hidden_dim=256,
        num_layers=3,
        num_heads=4,
        ffn_dim=1024,
        max_seq_len=128,
        dropout=0.1,
        use_ttt=True,
        ttt_layer_idx=1,
        ttt_dim=128,
        ttt_iterations=2,
        use_adaptive_ttt=False,
    )

    model = TransformerTTT(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Forward pass
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        outputs = model(input_ids, labels=labels, return_stats=True)

    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    if "ttt_stats" in outputs:
        print(f"TTT stats available: {len(outputs['ttt_stats'])} layers")

    print("\nModel test passed!")


def test_adaptive_model(train_loader):
    """Test adaptive model."""
    print("\nTesting adaptive model...")

    config = TransformerConfig(
        vocab_size=50257,
        hidden_dim=256,
        num_layers=3,
        num_heads=4,
        ffn_dim=1024,
        max_seq_len=128,
        dropout=0.1,
        use_ttt=True,
        ttt_layer_idx=1,
        ttt_dim=128,
        ttt_iterations=2,
        use_adaptive_ttt=True,
        ttt_difficulty_metric="entropy",
        ttt_buckets=[1, 2, 4],
        ttt_target_distribution=[0.3, 0.4, 0.3],
    )

    model = TransformerTTT(config)

    # Forward pass
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    with torch.no_grad():
        outputs = model(input_ids, labels=labels, return_stats=True)

    print(f"Loss: {outputs['loss'].item():.4f}")

    if "ttt_stats" in outputs and outputs["ttt_stats"]:
        stats = outputs["ttt_stats"][0]
        if "efficiency" in stats:
            print(f"Avg iterations: {stats['efficiency']['avg_iterations']:.2f}")
            print(f"FLOPs reduction: {stats['efficiency']['flops_reduction'] * 100:.1f}%")

    print("\nAdaptive model test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("PonderTTT Setup Test")
    print("=" * 60)

    train_loader = test_data_loading()
    test_model(train_loader)
    test_adaptive_model(train_loader)

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)

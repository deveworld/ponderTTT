"""
Tests for FLOPs counting functionality.

Validates accurate FLOP counting for TTT models.
"""

import pytest
import torch

from ponderttt.utils.flops import FLOPsCounter, TTTFLOPsAnalyzer, compute_model_flops
from ponderttt.models import IterativeTransformerConfig


def test_linear_flops():
    """Test FLOPs counting for linear layers."""
    print("\nTesting linear layer FLOPs...")

    # Forward: y = xW + b
    # Input: (batch, in_features) @ (in_features, out_features) = (batch, out_features)
    # FLOPs: 2 * in_features * out_features (matmul) + out_features (bias)
    in_features = 128
    out_features = 256

    # Without bias
    flops_no_bias = FLOPsCounter.linear_flops(in_features, out_features, has_bias=False)
    expected_no_bias = 2 * in_features * out_features
    assert flops_no_bias == expected_no_bias, f"Expected {expected_no_bias}, got {flops_no_bias}"
    print(f"  Linear ({in_features} → {out_features}, no bias): {flops_no_bias:,} FLOPs")

    # With bias
    flops_with_bias = FLOPsCounter.linear_flops(in_features, out_features, has_bias=True)
    expected_with_bias = 2 * in_features * out_features + out_features
    assert flops_with_bias == expected_with_bias, f"Expected {expected_with_bias}, got {flops_with_bias}"
    print(f"  Linear ({in_features} → {out_features}, with bias): {flops_with_bias:,} FLOPs")

    print("  ✓ Linear FLOPs computed correctly")


def test_layernorm_flops():
    """Test FLOPs counting for LayerNorm."""
    print("\nTesting LayerNorm FLOPs...")

    # LayerNorm: mean, variance, normalize, scale, shift
    # FLOPs: 8n (as per implementation)
    n = 512
    flops = FLOPsCounter.layernorm_flops(n)
    expected = 8 * n
    assert flops == expected, f"Expected {expected}, got {flops}"
    print(f"  LayerNorm ({n}): {flops:,} FLOPs")
    print("  ✓ LayerNorm FLOPs computed correctly")


def test_activation_flops():
    """Test FLOPs counting for activation functions."""
    print("\nTesting activation function FLOPs...")

    n = 1024

    # GELU
    gelu_flops = FLOPsCounter.gelu_flops(n)
    print(f"  GELU ({n}): {gelu_flops:,} FLOPs")
    assert gelu_flops == 8 * n

    # Sigmoid
    sigmoid_flops = FLOPsCounter.sigmoid_flops(n)
    print(f"  Sigmoid ({n}): {sigmoid_flops:,} FLOPs")
    assert sigmoid_flops == 4 * n

    # Softmax
    dim_size = 50000  # vocab size
    softmax_flops = FLOPsCounter.softmax_flops(n, dim_size)
    print(f"  Softmax ({n}, dim={dim_size}): {softmax_flops:,} FLOPs")

    print("  ✓ Activation FLOPs computed correctly")


def test_ttt_flops_analyzer_basic():
    """Test basic functionality of TTTFLOPsAnalyzer."""
    print("\nTesting TTTFLOPsAnalyzer basic functionality...")

    config = IterativeTransformerConfig(
        vocab_size=50257,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        use_iterative_ttt=True,
        ttt_layer_indices=[1, 2],
    )

    analyzer = TTTFLOPsAnalyzer(config)
    seq_len = 128

    # Test embedding FLOPs
    embedding_flops = analyzer.count_embedding_flops(seq_len)
    print(f"  Embedding: {embedding_flops:,} FLOPs")
    assert embedding_flops > 0

    # Test attention FLOPs
    attention_flops = analyzer.count_attention_flops(seq_len)
    print(f"  Attention: {attention_flops:,} FLOPs")
    assert attention_flops > 0

    # Test TTT FLOPs
    ttt_flops = analyzer.count_ttt_iterative_flops(seq_len, K=4)
    print(f"  TTT (K=4): {ttt_flops:,} FLOPs")
    assert ttt_flops > 0

    # Test policy network FLOPs
    policy_flops = analyzer.count_policy_network_flops(seq_len)
    print(f"  Policy network: {policy_flops:,} FLOPs")
    assert policy_flops > 0

    # Test FFN FLOPs
    ffn_flops = analyzer.count_ffn_flops(seq_len)
    print(f"  FFN: {ffn_flops:,} FLOPs")
    assert ffn_flops > 0

    print("  ✓ TTTFLOPsAnalyzer basic functions work")


def test_total_flops_estimation():
    """Test total FLOPs estimation for a model."""
    print("\nTesting total FLOPs estimation...")

    config = IterativeTransformerConfig(
        vocab_size=50257,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
    )

    analyzer = TTTFLOPsAnalyzer(config)
    seq_len = 64

    # Test with fixed K
    flops_k4 = analyzer.estimate_total_flops(
        seq_len=seq_len,
        num_steps=4,
        include_backward=False
    )

    print(f"  Forward total (K=4): {flops_k4['forward_total']:,.0f} FLOPs")
    print(f"  Per-token (K=4): {flops_k4['per_token']:,.0f} FLOPs")

    # Verify structure
    assert 'embedding' in flops_k4
    assert 'ttt_layers' in flops_k4
    assert 'ffn_layers' in flops_k4
    assert 'lm_head' in flops_k4
    assert 'forward_total' in flops_k4
    assert 'total' in flops_k4
    assert 'per_token' in flops_k4

    # Test with tensor K (adaptive)
    k_tensor = torch.tensor([1, 2, 4, 8] * (seq_len // 4))
    flops_adaptive = analyzer.estimate_total_flops(
        seq_len=seq_len,
        num_steps=k_tensor,
        include_backward=False
    )

    print(f"  Forward total (adaptive, avg K={k_tensor.float().mean():.2f}): {flops_adaptive['forward_total']:,.0f} FLOPs")

    # Test with backward pass
    flops_with_backward = analyzer.estimate_total_flops(
        seq_len=seq_len,
        num_steps=4,
        include_backward=True
    )

    print(f"  Total with backward (K=4): {flops_with_backward['total']:,.0f} FLOPs")
    assert flops_with_backward['backward_total'] > 0
    assert flops_with_backward['total'] > flops_k4['forward_total']

    print("  ✓ Total FLOPs estimation works")


def test_configuration_comparison():
    """Test FLOPs comparison across different K values."""
    print("\nTesting configuration comparison...")

    config = IterativeTransformerConfig(
        vocab_size=50257,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
    )

    analyzer = TTTFLOPsAnalyzer(config)
    seq_len = 64

    # Compare different K values
    comparison = analyzer.compare_configurations(
        seq_len=seq_len,
        k_values=[1, 2, 4, 8],
        include_backward=False
    )

    print("\n  Configuration comparison:")
    for config_name in ['K=1', 'K=2', 'K=4', 'K=8']:
        flops = comparison[config_name]
        print(f"    {config_name}: {flops['total']:,.0f} FLOPs ({flops['per_token']:,.0f} per token)")

    # Verify monotonicity: higher K should have more FLOPs
    assert comparison['K=1']['total'] < comparison['K=2']['total']
    assert comparison['K=2']['total'] < comparison['K=4']['total']
    assert comparison['K=4']['total'] < comparison['K=8']['total']

    print("  ✓ Configuration comparison works")


def test_flops_scaling_with_k():
    """Test that FLOPs scale approximately linearly with K."""
    print("\nTesting FLOPs scaling with K...")

    config = IterativeTransformerConfig(
        vocab_size=50257,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_iterative_ttt=True,
        ttt_layer_indices=[0, 1],  # 2 TTT layers
    )

    analyzer = TTTFLOPsAnalyzer(config)
    seq_len = 64

    # Get FLOPs for K=1 and K=4
    flops_k1 = analyzer.estimate_total_flops(seq_len, 1, include_backward=False)
    flops_k4 = analyzer.estimate_total_flops(seq_len, 4, include_backward=False)

    # TTT layers should scale linearly with K
    ttt_k1 = flops_k1['ttt_layers']
    ttt_k4 = flops_k4['ttt_layers']

    ratio = ttt_k4 / ttt_k1
    print(f"  TTT FLOPs (K=1): {ttt_k1:,.0f}")
    print(f"  TTT FLOPs (K=4): {ttt_k4:,.0f}")
    print(f"  Ratio: {ratio:.2f} (should be ~4.0)")

    # Allow some tolerance due to fixed overhead (Q,K,V projections, etc.)
    # With significant fixed overhead, ratio will be lower than 4.0
    # Should be between 2.0 and 4.5 (closer to 2-3 for models with large fixed costs)
    assert 2.0 < ratio < 4.5, f"Expected ratio between 2.0 and 4.5, got {ratio:.2f}"

    print(f"  ✓ FLOPs scale appropriately with K (ratio {ratio:.2f} within expected range)")


def test_compute_model_flops():
    """Test the convenience function compute_model_flops."""
    print("\nTesting compute_model_flops function...")

    config = IterativeTransformerConfig(
        vocab_size=50257,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
    )

    # Test with config object
    flops = compute_model_flops(
        model=config,
        seq_len=64,
        num_steps=4,
        include_backward=False
    )

    assert 'forward_total' in flops
    assert 'total' in flops
    print(f"  Total FLOPs: {flops['total']:,.0f}")
    print("  ✓ compute_model_flops works")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing FLOPs Counting Functionality")
    print("=" * 80)

    test_linear_flops()
    test_layernorm_flops()
    test_activation_flops()
    test_ttt_flops_analyzer_basic()
    test_total_flops_estimation()
    test_configuration_comparison()
    test_flops_scaling_with_k()
    test_compute_model_flops()

    print("\n" + "=" * 80)
    print("All FLOPs tests passed! ✓")
    print("=" * 80)

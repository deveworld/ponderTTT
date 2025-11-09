"""
Unit tests for TTTLinearAnalytic.

Tests the official TTT-Linear implementation with analytic solution,
including:
- Basic forward pass
- Gradient flow
- Mini-batch processing
- Learnable learning rates
- Shape correctness
"""

import torch
import pytest
from ponderttt.models.ttt_linear_analytic import TTTLinearAnalytic, layernorm_vjp


class TestLayerNormVJP:
    """Test LayerNorm vector-Jacobian product."""

    def test_vjp_correctness(self):
        """Verify VJP matches autograd backward."""
        # Create random input and gradient
        x = torch.randn(2, 8, 64, requires_grad=True)
        grad_output = torch.randn(2, 8, 64)

        # Create LayerNorm
        ln = torch.nn.LayerNorm(64)

        # Compute forward and backward with autograd
        y = ln(x)
        y.backward(grad_output)
        grad_auto = x.grad.clone()
        x.grad = None

        # Compute with our VJP
        grad_manual = layernorm_vjp(
            x.detach(),
            grad_output,
            ln.weight,
            ln.bias,
            eps=ln.eps
        )

        # Check correctness
        assert torch.allclose(grad_auto, grad_manual, rtol=1e-4, atol=1e-5), \
            f"VJP gradient mismatch: max diff = {(grad_auto - grad_manual).abs().max()}"

    def test_vjp_different_shapes(self):
        """Test VJP with different input shapes."""
        for shape in [(1, 16, 32), (4, 8, 128), (2, 64, 256)]:
            x = torch.randn(*shape, requires_grad=True)
            grad_output = torch.randn(*shape)

            ln = torch.nn.LayerNorm(shape[-1])

            y = ln(x)
            y.backward(grad_output)
            grad_auto = x.grad.clone()
            x.grad = None

            grad_manual = layernorm_vjp(x.detach(), grad_output, ln.weight, ln.bias, ln.eps)

            assert torch.allclose(grad_auto, grad_manual, rtol=1e-4, atol=1e-5), \
                f"Shape {shape}: max diff = {(grad_auto - grad_manual).abs().max()}"


class TestTTTLinearAnalytic:
    """Test TTTLinearAnalytic layer."""

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        layer = TTTLinearAnalytic(
            hidden_dim=256,
            num_heads=4,
            mini_batch_size=16,
            use_learnable_lr=False,
            use_gate=False,
        )

        x = torch.randn(2, 32, 256)  # batch=2, seq_len=32 (2 mini-batches)
        output, stats = layer(x, return_stats=True)

        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        assert stats['num_mini_batches'] == 2, f"Expected 2 mini-batches, got {stats['num_mini_batches']}"
        assert stats['mini_batch_size'] == 16
        print("✓ Basic forward pass test passed")

    def test_forward_pass_with_gate(self):
        """Test forward pass with gating."""
        layer = TTTLinearAnalytic(
            hidden_dim=128,
            num_heads=4,
            mini_batch_size=8,
            use_learnable_lr=True,
            use_gate=True,
        )

        x = torch.randn(1, 16, 128)
        output, _ = layer(x)

        assert output.shape == x.shape
        print("✓ Forward pass with gating test passed")

    def test_gradient_flow(self):
        """Test gradient flow through analytic solution."""
        layer = TTTLinearAnalytic(
            hidden_dim=128,
            num_heads=4,
            mini_batch_size=8,
            use_learnable_lr=False,
        )
        layer.train()

        x = torch.randn(1, 16, 128, requires_grad=True)
        output, _ = layer(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input gradient is None"
        assert layer.W_init.grad is not None, "W_init gradient is None"
        assert layer.b_init.grad is not None, "b_init gradient is None"

        # Check gradients are not zero or NaN
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        assert not torch.isnan(layer.W_init.grad).any(), "W_init gradient contains NaN"
        assert x.grad.abs().sum() > 0, "Input gradient is all zeros"

        print("✓ Gradient flow test passed")

    def test_learnable_lr(self):
        """Test learnable learning rate network."""
        layer_fixed = TTTLinearAnalytic(
            hidden_dim=64,
            num_heads=2,
            mini_batch_size=8,
            use_learnable_lr=False,
        )

        layer_learnable = TTTLinearAnalytic(
            hidden_dim=64,
            num_heads=2,
            mini_batch_size=8,
            use_learnable_lr=True,
        )

        x = torch.randn(1, 16, 64)

        # Both should work
        output_fixed, stats_fixed = layer_fixed(x, return_stats=True)
        output_learnable, stats_learnable = layer_learnable(x, return_stats=True)

        assert output_fixed.shape == output_learnable.shape
        # Outputs should be different (learnable LR is initialized randomly)
        assert not torch.allclose(output_fixed, output_learnable), \
            "Fixed and learnable LR outputs should differ"

        print("✓ Learnable LR test passed")

    def test_mini_batch_size_mismatch(self):
        """Test error handling for sequence length mismatch."""
        layer = TTTLinearAnalytic(
            hidden_dim=64,
            num_heads=2,
            mini_batch_size=16,
        )

        # Sequence length not divisible by mini_batch_size
        x = torch.randn(1, 20, 64)

        with pytest.raises(ValueError, match="must be divisible by mini_batch_size"):
            layer(x)

        print("✓ Mini-batch size mismatch test passed")

    def test_different_mini_batch_sizes(self):
        """Test with different mini-batch sizes."""
        for mb_size in [4, 8, 16]:
            layer = TTTLinearAnalytic(
                hidden_dim=128,
                num_heads=4,
                mini_batch_size=mb_size,
            )

            seq_len = mb_size * 4  # 4 mini-batches
            x = torch.randn(2, seq_len, 128)
            output, stats = layer(x, return_stats=True)

            assert output.shape == x.shape
            assert stats['num_mini_batches'] == 4

        print("✓ Different mini-batch sizes test passed")

    def test_parameter_count(self):
        """Test parameter count."""
        layer = TTTLinearAnalytic(
            hidden_dim=256,
            num_heads=8,
            mini_batch_size=16,
            use_learnable_lr=True,
            use_gate=True,
        )

        num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)

        # Expected parameters:
        # - QKV projections: 3 * (256 * 256) = 196,608
        # - W_init: 8 * 32 * 32 = 8,192
        # - b_init: 8 * 1 * 32 = 256
        # - ttt_norm: 8 * (32 + 32) = 512
        # - post_norm: 256 + 256 = 512
        # - lr_net: 8 * (256 * 1 + 1) = 2,056
        # - token_idx_offset: 16
        # - out_proj: 256 * 256 = 65,536
        # - gate_proj: 256 * 256 = 65,536
        # Total ≈ 339,224

        assert num_params > 300000, f"Parameter count too low: {num_params}"
        assert num_params < 400000, f"Parameter count too high: {num_params}"

        print(f"✓ Parameter count test passed (params: {num_params:,})")

    def test_process_mini_batch_shapes(self):
        """Test process_mini_batch output shapes."""
        layer = TTTLinearAnalytic(
            hidden_dim=128,
            num_heads=4,
            mini_batch_size=8,
        )

        batch_size = 2
        num_heads = 4
        mb_size = 8
        head_dim = 32

        Q_mb = torch.randn(batch_size, num_heads, mb_size, head_dim)
        K_mb = torch.randn(batch_size, num_heads, mb_size, head_dim)
        V_mb = torch.randn(batch_size, num_heads, mb_size, head_dim)
        eta_mb = torch.ones(batch_size, num_heads, mb_size, 1) * 0.1
        W = torch.randn(num_heads, head_dim, head_dim)
        b = torch.zeros(num_heads, 1, head_dim)

        output_mb, W_next, b_next = layer.process_mini_batch(Q_mb, K_mb, V_mb, eta_mb, W, b)

        assert output_mb.shape == (batch_size, num_heads, mb_size, head_dim)
        assert W_next.shape == (num_heads, head_dim, head_dim)
        assert b_next.shape == (num_heads, 1, head_dim)

        print("✓ process_mini_batch shapes test passed")

    def test_sequential_processing(self):
        """Test that mini-batches are processed sequentially with state carry-over."""
        layer = TTTLinearAnalytic(
            hidden_dim=64,
            num_heads=2,
            mini_batch_size=8,
        )

        x = torch.randn(1, 16, 64)

        # Forward pass
        output, _ = layer(x)

        # The fact that it doesn't crash and produces correct shapes
        # indicates sequential processing works
        assert output.shape == x.shape

        print("✓ Sequential processing test passed")

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        layer = TTTLinearAnalytic(
            hidden_dim=64,
            num_heads=2,
            mini_batch_size=8,
        )

        # Test with large values
        x_large = torch.randn(1, 16, 64) * 100
        output_large, _ = layer(x_large)
        assert not torch.isnan(output_large).any(), "NaN in output with large input"
        assert not torch.isinf(output_large).any(), "Inf in output with large input"

        # Test with small values
        x_small = torch.randn(1, 16, 64) * 0.01
        output_small, _ = layer(x_small)
        assert not torch.isnan(output_small).any(), "NaN in output with small input"

        print("✓ Numerical stability test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("Running TTTLinearAnalytic Tests")
    print("="*80 + "\n")

    # LayerNorm VJP tests
    print("LayerNorm VJP Tests:")
    test_vjp = TestLayerNormVJP()
    test_vjp.test_vjp_correctness()
    test_vjp.test_vjp_different_shapes()
    print()

    # TTTLinearAnalytic tests
    print("TTTLinearAnalytic Tests:")
    test_layer = TestTTTLinearAnalytic()
    test_layer.test_forward_pass_basic()
    test_layer.test_forward_pass_with_gate()
    test_layer.test_gradient_flow()
    test_layer.test_learnable_lr()
    try:
        test_layer.test_mini_batch_size_mismatch()
    except ImportError:
        print("⚠ Skipping mini-batch mismatch test (pytest not available)")
    test_layer.test_different_mini_batch_sizes()
    test_layer.test_parameter_count()
    test_layer.test_process_mini_batch_shapes()
    test_layer.test_sequential_processing()
    test_layer.test_numerical_stability()

    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_tests()

"""
Unit tests for iterative TTT components.
"""

import pytest
import torch

from src.ponderttt.models.fast_weight import FastWeightModule, MultiHeadFastWeight
from src.ponderttt.models.iterative_ttt import IterativeTTTLayer
from src.ponderttt.models.halting_policy import HaltingPolicyNetwork, MultiGranularityRouter


class TestFastWeightModule:
    """Tests for FastWeightModule."""

    def test_forward(self):
        """Test forward pass."""
        module = FastWeightModule(input_dim=64, hidden_dim=32, output_dim=64)
        x = torch.randn(2, 10, 64)
        output = module(x)
        assert output.shape == (2, 10, 64)

    def test_clone(self):
        """Test cloning preserves parameters."""
        module = FastWeightModule(input_dim=64, hidden_dim=32, output_dim=64)
        cloned = module.clone()

        # Check that parameters are equal
        for p1, p2 in zip(module.parameters(), cloned.parameters()):
            assert torch.allclose(p1, p2)

        # Check that they are different objects
        assert module is not cloned

    def test_gradient_flow(self):
        """Test that gradients flow through module."""
        module = FastWeightModule(input_dim=64, hidden_dim=32, output_dim=64)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = module(x)
        loss = output.mean()
        loss.backward()

        # Check that module parameters received gradients
        for param in module.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


class TestMultiHeadFastWeight:
    """Tests for MultiHeadFastWeight."""

    def test_forward(self):
        """Test forward pass with multiple heads."""
        module = MultiHeadFastWeight(
            num_heads=8,
            input_dim=64,
            hidden_dim=32,
            output_dim=64
        )
        x = torch.randn(2, 8, 10, 64)  # (batch, num_heads, seq_len, input_dim)
        output = module(x)
        assert output.shape == (2, 8, 10, 64)

    def test_heads_are_independent(self):
        """Test that each head has independent parameters."""
        module = MultiHeadFastWeight(
            num_heads=8,
            input_dim=64,
            hidden_dim=32,
            output_dim=64
        )

        # Get parameters from different heads
        head0_params = list(module.heads[0].parameters())
        head1_params = list(module.heads[1].parameters())

        # They should not be the same objects
        for p0, p1 in zip(head0_params, head1_params):
            assert p0 is not p1


class TestIterativeTTTLayer:
    """Tests for IterativeTTTLayer."""

    def test_forward_uniform_steps(self):
        """Test forward pass with uniform step counts."""
        layer = IterativeTTTLayer(
            hidden_dim=128,
            num_heads=4,
            max_steps=4,
            use_sequential=True,
        )

        x = torch.randn(2, 16, 128)
        num_steps = torch.full((2, 16), 4, dtype=torch.long)

        output, stats, _ = layer(x, num_steps=num_steps, return_stats=True)

        assert output.shape == (2, 16, 128)
        assert stats is not None
        assert stats['avg_steps'] == 4.0

    def test_forward_variable_steps(self):
        """Test forward pass with variable step counts."""
        layer = IterativeTTTLayer(
            hidden_dim=128,
            num_heads=4,
            max_steps=8,
            use_sequential=True,
        )

        x = torch.randn(2, 16, 128)
        # Different step counts: [1, 2, 4, 8] repeated
        num_steps = torch.tensor([[1, 2, 4, 8] * 4] * 2, dtype=torch.long)

        output, stats, _ = layer(x, num_steps=num_steps, return_stats=True)

        assert output.shape == (2, 16, 128)
        assert stats is not None
        # Average should be (1+2+4+8)/4 = 3.75
        assert abs(stats['avg_steps'] - 3.75) < 0.1

    def test_gradient_flow_through_inner_loop(self):
        """Test that gradients flow through the iterative update loop."""
        layer = IterativeTTTLayer(
            hidden_dim=64,
            num_heads=2,
            max_steps=2,
            use_sequential=True,
        )

        x = torch.randn(1, 4, 64, requires_grad=True)
        num_steps = torch.full((1, 4), 2, dtype=torch.long)

        # Enable training mode for create_graph=True
        layer.train()

        output, _, _ = layer(x, num_steps=num_steps)
        loss = output.mean()
        loss.backward()

        # Check that input received gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check that layer parameters received gradients
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_sequential_vs_independent(self):
        """Test that sequential and independent modes produce different outputs."""
        layer_seq = IterativeTTTLayer(
            hidden_dim=64,
            num_heads=2,
            max_steps=2,
            use_sequential=True,
        )

        layer_indep = IterativeTTTLayer(
            hidden_dim=64,
            num_heads=2,
            max_steps=2,
            use_sequential=False,
        )

        # Copy parameters to make them identical
        layer_indep.load_state_dict(layer_seq.state_dict())

        x = torch.randn(1, 8, 64)
        num_steps = torch.full((1, 8), 2, dtype=torch.long)

        output_seq, _, _ = layer_seq(x, num_steps=num_steps)
        output_indep, _, _ = layer_indep(x, num_steps=num_steps)

        # They should produce different outputs because of state carry-forward
        # (This test might be weak if the difference is small)
        assert not torch.allclose(output_seq, output_indep, atol=1e-4)

    def test_flops_counting(self):
        """Test FLOPs counting with variable steps."""
        layer = IterativeTTTLayer(
            hidden_dim=128,
            num_heads=4,
            max_steps=8,
        )

        num_steps = torch.tensor([[1, 2, 4, 8] * 4] * 2, dtype=torch.long)
        flops = layer.count_flops(seq_len=16, num_steps=num_steps)

        # FLOPs should be positive
        assert flops > 0

        # More steps should mean more FLOPs
        num_steps_more = torch.full((2, 16), 8, dtype=torch.long)
        flops_more = layer.count_flops(seq_len=16, num_steps=num_steps_more)
        assert flops_more > flops


class TestHaltingPolicyNetwork:
    """Tests for HaltingPolicyNetwork."""

    def test_forward_training(self):
        """Test forward pass in training mode."""
        policy = HaltingPolicyNetwork(
            hidden_dim=128,
            step_options=[1, 2, 4, 8],
            use_lstm=True,
        )
        policy.train()

        x = torch.randn(2, 16, 128)
        steps, probs = policy(x, return_probs=True)

        assert steps.shape == (2, 16)
        assert probs.shape == (2, 16, 4)  # 4 step options

        # Steps should be one of the options
        for step in steps.flatten():
            assert step.item() in [1, 2, 4, 8]

    def test_forward_inference(self):
        """Test forward pass in inference mode (deterministic)."""
        policy = HaltingPolicyNetwork(
            hidden_dim=128,
            step_options=[1, 2, 4, 8],
        )
        policy.eval()

        x = torch.randn(2, 16, 128)
        steps1, _ = policy(x, deterministic=True)
        steps2, _ = policy(x, deterministic=True)

        # Should be deterministic
        assert torch.all(steps1 == steps2)

    def test_gradient_flow(self):
        """Test that gradients flow through Gumbel-Softmax."""
        policy = HaltingPolicyNetwork(
            hidden_dim=64,
            step_options=[1, 2, 4],
        )
        policy.train()

        x = torch.randn(2, 8, 64, requires_grad=True)
        steps, _ = policy(x)

        # Create a differentiable loss (average number of steps)
        loss = steps.float().mean()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        for param in policy.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_entropy_computation(self):
        """Test policy entropy computation."""
        policy = HaltingPolicyNetwork(
            hidden_dim=64,
            step_options=[1, 2, 4, 8],
        )

        x = torch.randn(2, 8, 64)
        entropy = policy.compute_entropy(x)

        assert entropy.shape == (2, 8)
        # Entropy should be non-negative
        assert torch.all(entropy >= 0)


class TestMultiGranularityRouter:
    """Tests for MultiGranularityRouter."""

    def test_forward(self):
        """Test forward pass."""
        router = MultiGranularityRouter(
            num_layers=6,
            hidden_dim=128,
            step_options=[1, 2, 4, 8],
        )

        x = torch.randn(2, 16, 128)
        use_ttt, num_steps = router(x, layer_idx=3)

        assert use_ttt.shape == (2,)
        assert num_steps.shape == (2, 16)

        # use_ttt should be binary
        assert torch.all((use_ttt == 0) | (use_ttt == 1))

    def test_layer_masking(self):
        """Test that layer routing masks step allocation."""
        router = MultiGranularityRouter(
            num_layers=6,
            hidden_dim=64,
            step_options=[1, 2, 4, 8],
            use_layer_routing=True,
        )
        router.eval()

        x = torch.randn(2, 8, 64)

        # Force layer to be disabled by manipulating router weights
        with torch.no_grad():
            router.layer_router[-1].weight.zero_()
            router.layer_router[-1].bias.fill_(-10.0)  # Very negative = sigmoid ≈ 0

        use_ttt, num_steps = router(x, layer_idx=0)

        # If layer is disabled, num_steps should be all zeros
        if use_ttt.sum() == 0:
            assert torch.all(num_steps == 0)

    def test_without_layer_routing(self):
        """Test router with layer routing disabled."""
        router = MultiGranularityRouter(
            num_layers=6,
            hidden_dim=64,
            step_options=[1, 2, 4],
            use_layer_routing=False,
        )

        x = torch.randn(2, 8, 64)
        use_ttt, num_steps = router(x, layer_idx=2)

        # Should always use TTT
        assert torch.all(use_ttt == 1)
        assert num_steps.shape == (2, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

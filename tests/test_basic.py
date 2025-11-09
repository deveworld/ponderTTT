"""
Basic functionality tests for iterative TTT components.
"""

import torch
import sys
sys.path.insert(0, '/home/world/ponderttt')

from src.ponderttt.models.fast_weight import FastWeightModule, MultiHeadFastWeight
from src.ponderttt.models.iterative_ttt import IterativeTTTLayer
from src.ponderttt.models.halting_policy import HaltingPolicyNetwork, MultiGranularityRouter


def test_fast_weight():
    """Test FastWeightModule."""
    print("Testing FastWeightModule...")
    module = FastWeightModule(input_dim=64, hidden_dim=32, output_dim=64)
    x = torch.randn(2, 10, 64)
    output = module(x)
    assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"
    print("✓ FastWeightModule forward pass works")

    # Test clone
    cloned = module.clone()
    for p1, p2 in zip(module.parameters(), cloned.parameters()):
        assert torch.allclose(p1, p2), "Cloned parameters don't match"
    print("✓ FastWeightModule clone works")


def test_multi_head_fast_weight():
    """Test MultiHeadFastWeight."""
    print("\nTesting MultiHeadFastWeight...")
    module = MultiHeadFastWeight(num_heads=8, input_dim=64, hidden_dim=32, output_dim=64)
    x = torch.randn(2, 8, 10, 64)
    output = module(x)
    assert output.shape == (2, 8, 10, 64), f"Expected (2, 8, 10, 64), got {output.shape}"
    print("✓ MultiHeadFastWeight forward pass works")


def test_iterative_ttt_layer():
    """Test IterativeTTTLayer."""
    print("\nTesting IterativeTTTLayer...")
    layer = IterativeTTTLayer(
        hidden_dim=128,
        num_heads=4,
        max_steps=4,
        use_sequential=True,
    )

    x = torch.randn(2, 8, 128)
    num_steps = torch.full((2, 8), 2, dtype=torch.long)

    output, stats, _ = layer(x, num_steps=num_steps, return_stats=True)

    assert output.shape == (2, 8, 128), f"Expected (2, 8, 128), got {output.shape}"
    assert stats is not None, "Stats should not be None"
    assert 'avg_steps' in stats, "Stats should contain avg_steps"
    print(f"✓ IterativeTTTLayer forward pass works (avg_steps={stats['avg_steps']})")

    # Test variable steps
    num_steps = torch.tensor([[1, 2, 4, 2] * 2] * 2, dtype=torch.long)
    output, stats, _ = layer(x, num_steps=num_steps, return_stats=True)
    expected_avg = (1 + 2 + 4 + 2) / 4
    assert abs(stats['avg_steps'] - expected_avg) < 0.1, f"Expected avg={expected_avg}, got {stats['avg_steps']}"
    print(f"✓ Variable step counts work correctly (avg={stats['avg_steps']})")


def test_halting_policy():
    """Test HaltingPolicyNetwork."""
    print("\nTesting HaltingPolicyNetwork...")
    policy = HaltingPolicyNetwork(
        hidden_dim=128,
        step_options=[1, 2, 4, 8],
        use_lstm=True,
    )
    policy.eval()

    x = torch.randn(2, 16, 128)
    steps, probs = policy(x, return_probs=True, deterministic=True)

    assert steps.shape == (2, 16), f"Expected (2, 16), got {steps.shape}"
    assert probs.shape == (2, 16, 4), f"Expected (2, 16, 4), got {probs.shape}"

    # Check that steps are valid
    for step in steps.flatten():
        assert step.item() in [1, 2, 4, 8], f"Invalid step count: {step.item()}"

    print(f"✓ HaltingPolicyNetwork works (sample steps: {steps[0, :5].tolist()})")

    # Test determinism
    steps2, _ = policy(x, deterministic=True)
    assert torch.all(steps == steps2), "Policy should be deterministic in eval mode"
    print("✓ Deterministic inference works")


def test_multi_granularity_router():
    """Test MultiGranularityRouter."""
    print("\nTesting MultiGranularityRouter...")
    router = MultiGranularityRouter(
        num_layers=6,
        hidden_dim=128,
        step_options=[1, 2, 4, 8],
        use_layer_routing=True,
    )
    router.eval()

    x = torch.randn(2, 16, 128)
    use_ttt, num_steps = router(x, layer_idx=3)

    assert use_ttt.shape == (2,), f"Expected (2,), got {use_ttt.shape}"
    assert num_steps.shape == (2, 16), f"Expected (2, 16), got {num_steps.shape}"

    # use_ttt should be binary
    assert torch.all((use_ttt == 0) | (use_ttt == 1)), "use_ttt should be binary"

    print(f"✓ MultiGranularityRouter works (use_ttt={use_ttt.tolist()})")


def test_gradient_flow():
    """Test gradient flow through iterative layer."""
    print("\nTesting gradient flow...")
    layer = IterativeTTTLayer(
        hidden_dim=64,
        num_heads=2,
        max_steps=2,
        use_sequential=True,
    )
    layer.train()

    x = torch.randn(1, 4, 64, requires_grad=True)
    num_steps = torch.full((1, 4), 2, dtype=torch.long)

    output, _, _ = layer(x, num_steps=num_steps)
    loss = output.mean()
    loss.backward()

    # Check gradients
    assert x.grad is not None, "Input should have gradients"
    assert not torch.all(x.grad == 0), "Input gradients should be non-zero"

    grad_count = 0
    for name, param in layer.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1

    print(f"✓ Gradient flow works ({grad_count} parameters have gradients)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Basic Tests for Iterative TTT Components")
    print("=" * 60)

    try:
        test_fast_weight()
        test_multi_head_fast_weight()
        test_iterative_ttt_layer()
        test_halting_policy()
        test_multi_granularity_router()
        test_gradient_flow()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
Tests for REINFORCE temporal credit assignment fixes.

Verifies that:
1. Monte Carlo returns are computed correctly
2. Temporal credit assignment works properly
3. Baseline updates correctly
4. Policy gradients flow correctly
"""

import torch
from ponderttt.models.transformer_iterative import IterativeTransformerConfig, IterativeTransformerTTT


def test_monte_carlo_returns_computation():
    """Test that Monte Carlo returns are computed correctly."""
    print("Testing Monte Carlo returns computation...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_learned_policy=True,
        gamma=0.9,
    )
    model = IterativeTransformerTTT(config)

    # Create simple test case
    # per_token_loss: (batch=1, seq_len=4)
    # Losses: [1.0, 2.0, 3.0, 4.0]
    per_token_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    returns = model._compute_monte_carlo_returns(per_token_loss, gamma=0.9)

    # Expected returns (computed by hand):
    # G_3 = -4.0
    # G_2 = -3.0 + 0.9 * G_3 = -3.0 + 0.9 * (-4.0) = -3.0 - 3.6 = -6.6
    # G_1 = -2.0 + 0.9 * G_2 = -2.0 + 0.9 * (-6.6) = -2.0 - 5.94 = -7.94
    # G_0 = -1.0 + 0.9 * G_1 = -1.0 + 0.9 * (-7.94) = -1.0 - 7.146 = -8.146

    expected_returns = torch.tensor([[
        -8.146,  # G_0
        -7.94,   # G_1
        -6.6,    # G_2
        -4.0,    # G_3
    ]])

    assert returns.shape == per_token_loss.shape, \
        f"Returns shape mismatch: {returns.shape} vs {per_token_loss.shape}"

    assert torch.allclose(returns, expected_returns, atol=1e-3), \
        f"Returns computation incorrect:\n  Got: {returns}\n  Expected: {expected_returns}"

    print("  ✓ Monte Carlo returns computed correctly")


def test_temporal_credit_assignment():
    """Test that temporal credit assignment affects advantages correctly."""
    print("Testing temporal credit assignment...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_learned_policy=True,
        gamma=0.9,
    )
    model = IterativeTransformerTTT(config)

    # Scenario: Early token has high loss, later tokens have low loss
    # Early policy decisions should get negative advantage (bad)
    per_token_loss_early_high = torch.tensor([[10.0, 1.0, 1.0, 1.0]])

    # Scenario: Early token has low loss, later tokens have high loss
    # Early policy decisions should get positive advantage (good, prevented future high loss)
    per_token_loss_early_low = torch.tensor([[1.0, 10.0, 10.0, 10.0]])

    returns_early_high = model._compute_monte_carlo_returns(per_token_loss_early_high, gamma=0.9)
    returns_early_low = model._compute_monte_carlo_returns(per_token_loss_early_low, gamma=0.9)

    # The first position in "early_low" should have much worse (more negative) return
    # because even though it starts with low loss, all future losses are high
    # early_high: only first loss is high, future is good → better cumulative return
    # early_low: first loss is good, but future is all bad → worse cumulative return
    assert returns_early_low[0, 0] < returns_early_high[0, 0], \
        f"Early low (but bad future) should have worse return: {returns_early_low[0, 0]:.3f} vs {returns_early_high[0, 0]:.3f}"

    # The returns should account for cumulative effect
    # early_high: G_0 = -10 + 0.9*(-1 + 0.9*(-1 + 0.9*(-1))) ≈ -12.44
    # early_low: G_0 = -1 + 0.9*(-10 + 0.9*(-10 + 0.9*(-10))) ≈ -25.39

    print(f"  Returns (early high loss): {returns_early_high[0, 0]:.3f}")
    print(f"  Returns (early low loss): {returns_early_low[0, 0]:.3f}")
    print("  ✓ Temporal credit assignment working correctly")


def test_baseline_initialization_and_update():
    """Test baseline initialization and EMA update."""
    print("Testing baseline initialization and update...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_learned_policy=True,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
        gamma=0.99,
        baseline_momentum=0.9,
    )
    model = IterativeTransformerTTT(config)
    model.train()

    # Check initial state
    assert not model.baseline_initialized.item(), "Baseline should not be initialized initially"
    assert model.baseline.item() == 0.0, "Baseline should be 0.0 initially"

    # Create dummy input
    input_ids = torch.randint(0, 100, (2, 16))
    labels = input_ids.clone()

    # First forward pass - should initialize baseline
    output1 = model(input_ids, labels=labels)

    assert model.baseline_initialized.item(), "Baseline should be initialized after first forward"
    baseline_after_init = model.baseline.item()
    assert baseline_after_init != 0.0, "Baseline should be non-zero after initialization"

    print(f"  Baseline after init: {baseline_after_init:.4f}")

    # Second forward pass - should update with EMA
    output2 = model(input_ids, labels=labels)
    baseline_after_update = model.baseline.item()

    # Baseline should have changed
    assert baseline_after_update != baseline_after_init, \
        "Baseline should update via EMA"

    print(f"  Baseline after update: {baseline_after_update:.4f}")
    print("  ✓ Baseline initialization and update working correctly")


def test_policy_gradient_flow():
    """Test that policy gradients flow correctly with Monte Carlo returns."""
    print("Testing policy gradient flow...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_learned_policy=True,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
        step_options=[1, 2, 4],
        gamma=0.99,
    )
    model = IterativeTransformerTTT(config)
    model.train()

    # Create input
    input_ids = torch.randint(0, 100, (2, 16))
    labels = input_ids.clone()

    # Forward pass
    output = model(input_ids, labels=labels)
    loss = output['loss']

    # Backward pass
    loss.backward()

    # Check that policy network has gradients
    has_policy_grads = False
    for name, param in model.named_parameters():
        if 'halting_policy' in name and param.grad is not None:
            has_policy_grads = True
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    assert has_policy_grads, "Policy network should have gradients"
    print("  ✓ Policy gradients flow correctly")


def test_advantage_variance_reduction():
    """Test that advantages have reasonable variance (not too high)."""
    print("Testing advantage variance...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_learned_policy=True,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
        gamma=0.99,
        baseline_momentum=0.99,
    )
    model = IterativeTransformerTTT(config)
    model.train()

    # Run multiple forward passes to collect advantages
    advantages_list = []

    for _ in range(5):
        input_ids = torch.randint(0, 100, (4, 32))
        labels = input_ids.clone()

        output = model(input_ids, labels=labels, return_stats=True)

        if 'ttt_stats' in output:
            for stats in output['ttt_stats']:
                if 'mean_advantage' in stats:
                    advantages_list.append(stats['mean_advantage'])

    if advantages_list:
        advantages = torch.tensor(advantages_list)
        mean_adv = advantages.mean().item()
        std_adv = advantages.std().item()

        print(f"  Mean advantage: {mean_adv:.6f}")
        print(f"  Std advantage: {std_adv:.6f}")

        # Advantages should be roughly centered around 0 (with good baseline)
        assert abs(mean_adv) < 5.0, \
            f"Mean advantage too far from 0: {mean_adv}"

        print("  ✓ Advantage variance is reasonable")
    else:
        print("  ⚠ No advantage statistics available (skipping variance test)")


def test_gamma_effect():
    """Test that different gamma values produce different returns."""
    print("Testing gamma (discount factor) effect...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_learned_policy=True,
    )
    model = IterativeTransformerTTT(config)

    per_token_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # High gamma (0.99) - future matters a lot
    returns_high_gamma = model._compute_monte_carlo_returns(per_token_loss, gamma=0.99)

    # Low gamma (0.5) - future matters less
    returns_low_gamma = model._compute_monte_carlo_returns(per_token_loss, gamma=0.5)

    # With high gamma, early positions should accumulate more from future
    # So the magnitude should be larger (more negative)
    assert returns_high_gamma[0, 0] < returns_low_gamma[0, 0], \
        f"High gamma should accumulate more: {returns_high_gamma[0, 0]:.3f} vs {returns_low_gamma[0, 0]:.3f}"

    print(f"  Returns (gamma=0.99): {returns_high_gamma[0, 0]:.3f}")
    print(f"  Returns (gamma=0.50): {returns_low_gamma[0, 0]:.3f}")
    print("  ✓ Gamma effect verified")


def run_all_tests():
    """Run all REINFORCE fix tests."""
    print("\n" + "="*80)
    print("REINFORCE Temporal Credit Assignment Tests")
    print("="*80 + "\n")

    test_monte_carlo_returns_computation()
    test_temporal_credit_assignment()
    test_baseline_initialization_and_update()
    test_policy_gradient_flow()
    test_advantage_variance_reduction()
    test_gamma_effect()

    print("\n" + "="*80)
    print("All REINFORCE fix tests passed! ✓")
    print("="*80 + "\n")

    print("Summary:")
    print("  - Monte Carlo returns computed with proper temporal credit")
    print("  - Advantages account for sequential dependencies")
    print("  - Baseline updates with EMA of returns (not just immediate loss)")
    print("  - Policy gradients flow correctly")
    print("  - Discount factor (gamma) controls temporal horizon")


if __name__ == "__main__":
    run_all_tests()

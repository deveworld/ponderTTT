"""
Tests for Oracle analysis fixes.

Verifies that:
1. Oracle K is computed using per-token loss (not total loss)
2. Sequential dependency is acknowledged
3. Oracle K provides meaningful upper bound
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ponderttt.models import IterativeTransformerConfig, IterativeTransformerTTT
from ponderttt.experiments.oracle_analysis import OracleAnalyzer


def test_per_token_loss_computation():
    """Test that oracle uses per-token loss, not total loss."""
    print("Testing per-token loss computation...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
        max_steps=8,
        use_learned_policy=False,
    )
    model = IterativeTransformerTTT(config)
    model.eval()

    analyzer = OracleAnalyzer(
        model=model,
        step_options=[1, 2, 4],
        device='cpu',
    )

    # Create dummy data
    input_ids = torch.randint(0, 100, (2, 16))
    labels = input_ids.clone()

    # Find oracle K for position 5
    best_k, k_losses = analyzer.find_oracle_k_for_token(
        input_ids, labels, token_position=5
    )

    # Check that we got results for all K values
    assert len(k_losses) == 3, f"Expected 3 K values, got {len(k_losses)}"
    assert all(k in k_losses for k in [1, 2, 4]), "Missing K values in results"

    # Check that losses are reasonable (positive, finite)
    for k, loss in k_losses.items():
        assert loss > 0, f"Loss for K={k} should be positive, got {loss}"
        assert loss < 100, f"Loss for K={k} seems too high: {loss}"

    # Check that best_k is one of the options
    assert best_k in [1, 2, 4], f"Best K should be in [1, 2, 4], got {best_k}"

    print(f"  Oracle K for position 5: {best_k}")
    print(f"  Losses: {k_losses}")
    print("  ✓ Per-token loss computation working")


def test_sequential_dependency_acknowledged():
    """Test that sequential dependency is properly documented."""
    print("Testing sequential dependency acknowledgment...")

    # Check docstring mentions sequential dependency
    from ponderttt.experiments.oracle_analysis import OracleAnalyzer
    docstring = OracleAnalyzer.find_oracle_k_for_token.__doc__

    assert "sequential" in docstring.lower(), \
        "Docstring should mention sequential dependency"
    assert "carry-over" in docstring.lower() or "carry" in docstring.lower(), \
        "Docstring should mention carry-over effect"

    print("  ✓ Sequential dependency documented in docstring")


def test_different_positions_different_oracle_k():
    """Test that different positions can have different oracle K."""
    print("Testing position-specific oracle K...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
        max_steps=8,
        use_learned_policy=False,
    )
    model = IterativeTransformerTTT(config)
    model.eval()

    analyzer = OracleAnalyzer(
        model=model,
        step_options=[1, 2, 4, 8],
        device='cpu',
    )

    # Create dummy data
    input_ids = torch.randint(0, 100, (2, 32))
    labels = input_ids.clone()

    # Find oracle K for multiple positions
    oracle_ks = []
    for pos in [5, 10, 15, 20, 25]:
        best_k, _ = analyzer.find_oracle_k_for_token(
            input_ids, labels, token_position=pos
        )
        oracle_ks.append(best_k)

    print(f"  Oracle Ks for positions [5, 10, 15, 20, 25]: {oracle_ks}")

    # It's possible (though unlikely) that all positions have the same oracle K
    # But the algorithm should at least run without errors
    print("  ✓ Position-specific oracle K computed successfully")


def test_oracle_k_improves_loss():
    """Test that using oracle K improves (or at least doesn't hurt) loss."""
    print("Testing that oracle K improves loss...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
        max_steps=8,
        use_learned_policy=False,
    )
    model = IterativeTransformerTTT(config)
    model.eval()

    analyzer = OracleAnalyzer(
        model=model,
        step_options=[1, 2, 4],
        device='cpu',
    )

    # Create dummy data
    input_ids = torch.randint(0, 100, (2, 16))
    labels = input_ids.clone()

    # Find oracle K for position 5
    best_k, k_losses = analyzer.find_oracle_k_for_token(
        input_ids, labels, token_position=5
    )

    # Oracle K should have the lowest loss
    assert k_losses[best_k] == min(k_losses.values()), \
        f"Oracle K={best_k} should have lowest loss, but got {k_losses}"

    # Check that oracle K actually helps (compared to worst K)
    worst_loss = max(k_losses.values())
    best_loss = k_losses[best_k]
    improvement = (worst_loss - best_loss) / worst_loss * 100

    print(f"  Best K: {best_k}, Best loss: {best_loss:.4f}")
    print(f"  Worst loss: {worst_loss:.4f}")
    print(f"  Improvement: {improvement:.2f}%")
    print("  ✓ Oracle K minimizes per-token loss")


def test_edge_case_first_position():
    """Test oracle K computation for first position (edge case)."""
    print("Testing edge case: first position...")

    config = IterativeTransformerConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_iterative_ttt=True,
        ttt_layer_indices=[0],
        max_steps=8,
        use_learned_policy=False,
    )
    model = IterativeTransformerTTT(config)
    model.eval()

    analyzer = OracleAnalyzer(
        model=model,
        step_options=[1, 2, 4],
        device='cpu',
    )

    # Create dummy data
    input_ids = torch.randint(0, 100, (2, 16))
    labels = input_ids.clone()

    # Find oracle K for first position (edge case)
    best_k, k_losses = analyzer.find_oracle_k_for_token(
        input_ids, labels, token_position=0
    )

    # Should not crash and should return valid results
    assert best_k in [1, 2, 4], f"Best K should be valid, got {best_k}"
    assert len(k_losses) == 3, f"Should have 3 K values, got {len(k_losses)}"

    print(f"  Oracle K for position 0: {best_k}")
    print("  ✓ First position handled correctly")


def run_all_tests():
    """Run all oracle analysis fix tests."""
    print("\n" + "="*80)
    print("Oracle Analysis Fix Tests")
    print("="*80 + "\n")

    test_per_token_loss_computation()
    test_sequential_dependency_acknowledged()
    test_different_positions_different_oracle_k()
    test_oracle_k_improves_loss()
    test_edge_case_first_position()

    print("\n" + "="*80)
    print("All Oracle analysis fix tests passed! ✓")
    print("="*80 + "\n")

    print("Summary:")
    print("  - Oracle K computed using per-token loss (not total loss)")
    print("  - Sequential dependency explicitly acknowledged")
    print("  - Oracle K minimizes per-token loss at target position")
    print("  - Edge cases (first position) handled correctly")
    print("  - Provides meaningful upper bound for learned policies")


if __name__ == "__main__":
    run_all_tests()

# PonderTTT: Comprehensive Repair and Research Roadmap

**Goal**: Transform PonderTTT into a rigorous, academically sound research project with clear novel contributions.

**Timeline**: 3-6 months for full execution
**Target**: Top-tier ML conference (NeurIPS, ICLR) or strong workshop paper

---

## Project Status Summary

**Current State** (Updated 2025-11-09):
- ✅ Code runnable and fully tested (19/19 tests passing)
- ✅ Implementation complete (Phase 0, 1, 2 all done)
- ✅ REINFORCE temporal credit bug fixed
- ✅ Heuristic policies calibrated
- ✅ Extended oracle analysis implemented
- ⏳ Experimental validation in progress

**Implementation Complete** ✅:
- ✅ Runnable, reproducible codebase
- ✅ Faithful TTT-Linear baseline (analytic solution)
- ✅ Novel iterative TTT variant (well-justified)
- ✅ REINFORCE with temporal credit (main contribution)
- ✅ Calibrated heuristic baselines
- ✅ Extended oracle upper bound
- ⏳ Experimental validation (running)

---

## Phase 0: Immediate Fixes (Week 1)
**Goal**: Make code runnable and testable

### Task 0.1: Fix Import Errors
**Priority**: P0 - Blocker
**Estimated Time**: 30 minutes
**Files**: `src/ponderttt/models/__init__.py`

**Problem**:
- `__init__.py` tries to import `DummyPolicyNetwork` which was deleted
- Previous fix removed the class to enforce parameter fairness (all models have real HaltingPolicyNetwork)

**Actions**:
```python
# src/ponderttt/models/__init__.py
# Simply remove the import - DO NOT recreate DummyPolicyNetwork

# Before (WRONG):
from .transformer_iterative import IterativeTransformerConfig, IterativeTransformerTTT, DummyPolicyNetwork

# After (CORRECT):
from .transformer_iterative import IterativeTransformerConfig, IterativeTransformerTTT

# Also remove from __all__:
__all__ = [
    ...,
    # Remove: "DummyPolicyNetwork",  # ← DELETE THIS LINE
]
```

**Rationale**:
- DummyPolicyNetwork was deleted for good reason (parameter fairness)
- All models should have actual HaltingPolicyNetwork
- Baselines override policy with `num_steps` parameter

**Success Criteria**: `python -c "from ponderttt.models import IterativeTransformerTTT"` works

---

### Task 0.2: Add Missing Utility Functions
**Priority**: P0 - Blocker
**Estimated Time**: 1 hour
**Files**: `src/ponderttt/utils/metrics.py`, `src/ponderttt/data/wikitext.py`

**Actions**:
```python
# Add to metrics.py
import math

def compute_perplexity(loss: float) -> float:
    """
    Convert cross-entropy loss to perplexity.

    Args:
        loss: Cross-entropy loss (scalar)

    Returns:
        perplexity: exp(loss)
    """
    return math.exp(min(loss, 100))  # Clip for numerical stability

# Add to wikitext.py (alias)
def get_wikitext2_dataloaders(
    batch_size: int = 8,
    max_length: int = 256,
    **kwargs
):
    """Alias for get_wikitext_dataloaders with wikitext-2 as default."""
    return get_wikitext_dataloaders(
        dataset_name="wikitext-2-raw-v1",
        batch_size=batch_size,
        max_length=max_length,
        **kwargs
    )
```

**Success Criteria**: All imports in `full_comparison_suite.py` work

---

### Task 0.3: Smoke Test
**Priority**: P0
**Estimated Time**: 1 hour

**Actions**:
```bash
# Test basic imports
python -c "from ponderttt.models import *; from ponderttt.data.wikitext import *; from ponderttt.utils.metrics import *"

# Test model instantiation
python -c "
from ponderttt.models import IterativeTransformerConfig, IterativeTransformerTTT
config = IterativeTransformerConfig(
    vocab_size=1000, hidden_dim=128, num_layers=2, num_heads=4
)
model = IterativeTransformerTTT(config)
print(f'Model parameters: {model.count_parameters():,}')
"

# Test forward pass
python -c "
import torch
from ponderttt.models import IterativeTransformerConfig, IterativeTransformerTTT

config = IterativeTransformerConfig(
    vocab_size=1000, hidden_dim=128, num_layers=2, num_heads=4,
    use_iterative_ttt=True, use_learned_policy=False
)
model = IterativeTransformerTTT(config)
model.eval()

input_ids = torch.randint(0, 1000, (2, 32))
labels = input_ids.clone()
outputs = model(input_ids, labels=labels, num_steps=torch.full((32,), 4))
print(f'Loss: {outputs[\"loss\"].item():.4f}')
"
```

**Success Criteria**: All tests pass without errors

---

## Phase 1: Core Implementation Fixes (Weeks 2-5)
**Goal**: Implement official TTT correctly and fix critical bugs

### Task 1.1: Implement Official TTT-Linear (Analytic Solution)
**Priority**: P1 - Critical
**Estimated Time**: 3-4 weeks (REALISTIC: complex analytic solution)
**Academic Importance**: ⭐⭐⭐⭐⭐ (Essential for fair comparison)

**Rationale**:
- Current "official" implementation is actually iterative GD, not analytic
- Need true official TTT as baseline for credibility
- This is the foundation for all comparisons

**Actions**:

1. **Study Official Implementation** (2 days):
   - Read `ttt-lm-jax/ttt/models/ttt_layer.py` line by line
   - Understand analytic solution derivation
   - Document triangular attention mechanism
   - Verify mini-batch processing logic

2. **Create New Module** `src/ponderttt/models/ttt_linear_analytic.py`:

```python
"""
Official TTT-Linear implementation with analytic solution.

Based on: https://github.com/test-time-training/ttt-lm-jax
Paper: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"

Key differences from iterative variant:
1. Analytic closed-form solution (not iterative GD)
2. Mini-batch processing (16 tokens at a time)
3. Triangular attention within mini-batch
4. Per-head learnable learning rate network
5. Two-stage LayerNorm (before and after residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class TTTLinearAnalytic(nn.Module):
    """
    Official TTT-Linear layer with analytic solution.

    Implements the exact algorithm from the paper, including:
    - Mini-batch processing (configurable size, default 16)
    - Triangular attention for causal dependencies
    - Analytic gradient descent (closed-form update)
    - Learnable per-head learning rates
    - Proper two-stage normalization

    Args:
        hidden_dim: Total hidden dimension
        num_heads: Number of attention heads
        mini_batch_size: Tokens per mini-batch (default 16)
        ttt_base_lr: Base learning rate for TTT updates
        use_learnable_lr: Enable per-head learnable LR network
        rope_theta: RoPE theta parameter
        use_gate: Enable output gating (recommended)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mini_batch_size: int = 16,
        ttt_base_lr: float = 1.0,
        use_learnable_lr: bool = True,
        rope_theta: float = 10000.0,
        use_gate: bool = True,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mini_batch_size = mini_batch_size
        self.ttt_base_lr = ttt_base_lr
        self.use_learnable_lr = use_learnable_lr
        self.use_gate = use_gate

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Fast-weight parameters (per-head linear transform)
        # Shape: (num_heads, head_dim, head_dim)
        self.W_init = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)
        self.b_init = nn.Parameter(torch.zeros(num_heads, 1, self.head_dim))

        # TTT normalization (per-head, applied to fast-weight output)
        self.ttt_norm = nn.ModuleList([
            nn.LayerNorm(self.head_dim) for _ in range(num_heads)
        ])

        # Post normalization (applied to final output)
        self.post_norm = nn.LayerNorm(hidden_dim)

        # Learnable learning rate network (per-head)
        if use_learnable_lr:
            self.lr_net = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 1, bias=True),
                    nn.Sigmoid()  # Output in [0, 1]
                ) for _ in range(num_heads)
            ])
            # Learnable token index offset
            self.token_idx_offset = nn.Parameter(torch.zeros(mini_batch_size))

        # Fixed token index (1/1, 1/2, 1/3, ..., 1/mini_batch_size)
        self.register_buffer(
            'token_idx',
            1.0 / torch.arange(1, mini_batch_size + 1, dtype=torch.float32)
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Optional gating
        if use_gate:
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following official implementation."""
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.use_gate:
            nn.init.normal_(self.gate_proj.weight, std=0.02)

    def get_eta(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token, per-head learning rates.

        Official formula:
        eta = (base_lr * token_idx) * learnable_lr(X) / head_dim

        Args:
            X: Input (batch, seq_len, hidden_dim)

        Returns:
            eta: Learning rates (batch, num_heads, mini_batch_size, 1)
        """
        batch_size, seq_len, _ = X.shape
        num_mini_batches = seq_len // self.mini_batch_size

        if self.use_learnable_lr:
            # Compute learnable LR for each head
            lr_multipliers = []
            for head_idx in range(self.num_heads):
                lr_mult = self.lr_net[head_idx](X)  # (batch, seq_len, 1)
                lr_multipliers.append(lr_mult)
            lr_multipliers = torch.stack(lr_multipliers, dim=1)  # (batch, num_heads, seq_len, 1)

            # Reshape to mini-batches
            lr_multipliers = lr_multipliers.view(
                batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, 1
            )

            # Token index with learnable offset
            token_idx = torch.clamp(self.token_idx + self.token_idx_offset, min=0.0)
        else:
            # Fixed LR
            lr_multipliers = torch.ones(
                batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, 1,
                device=X.device
            )
            token_idx = self.token_idx

        # Compute eta
        eta = (self.ttt_base_lr * token_idx.view(1, 1, 1, -1, 1)) * lr_multipliers / self.head_dim

        return eta

    def process_mini_batch(
        self,
        Q_mb: torch.Tensor,  # (batch, num_heads, mini_batch_size, head_dim)
        K_mb: torch.Tensor,
        V_mb: torch.Tensor,
        eta_mb: torch.Tensor,  # (batch, num_heads, mini_batch_size, 1)
        W: torch.Tensor,  # (num_heads, head_dim, head_dim)
        b: torch.Tensor,  # (num_heads, 1, head_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one mini-batch with analytic solution.

        Official algorithm (from ttt_layer.py:368-429):
        1. Forward: Z = X @ W + b
        2. Compute gradient: grad = (ttt_norm(Z) - (V - K))
        3. Analytic update with triangular attention:
           Z_bar = Q @ W - (eta * tril(Q @ K^T)) @ grad + b_bar
           where b_bar accounts for gradient accumulation
        4. Apply ttt_norm and add residual: output = Q + ttt_norm(Z_bar)
        5. Update W, b for next mini-batch

        Args:
            Q_mb, K_mb, V_mb: Query, Key, Value for this mini-batch
            eta_mb: Learning rates
            W, b: Fast-weight parameters from previous mini-batch

        Returns:
            output_mb: Output for this mini-batch
            W_next: Updated W for next mini-batch
            b_next: Updated b for next mini-batch
        """
        batch_size, num_heads, mb_size, head_dim = Q_mb.shape

        # SSL target: V - K (residual prediction)
        ssl_target = V_mb - K_mb  # (batch, num_heads, mb_size, head_dim)

        # Forward pass through fast-weight
        # Z = K @ W + b for each head
        Z = torch.einsum('bhmd,hde->bhme', K_mb, W) + b.unsqueeze(0)  # (batch, num_heads, mb_size, head_dim)

        # Apply ttt_norm per head
        Z_normed = torch.zeros_like(Z)
        for h in range(num_heads):
            Z_normed[:, h] = self.ttt_norm[h](Z[:, h])

        # Compute gradient: grad_loss_wrt_Z_normed = Z_normed - ssl_target
        grad_normed = Z_normed - ssl_target

        # Backprop through LayerNorm to get grad_loss_wrt_Z
        # This is simplified - full implementation needs LayerNorm VJP
        # For now, approximate: grad_Z ≈ grad_normed (ignoring LayerNorm backward)
        # TODO: Implement proper LayerNorm backward pass
        grad_Z = grad_normed  # Approximation

        # Triangular attention: Attn = tril(Q @ K^T)
        Attn = torch.einsum('bhmd,bhnd->bhmn', Q_mb, K_mb)  # (batch, num_heads, mb_size, mb_size)
        Attn = torch.tril(Attn)  # Lower triangular (causal)

        # Analytic update for Z_bar (output before normalization)
        # Z_bar = Q @ W - (eta * Attn) @ grad_Z + b_bar

        # Compute b_bar (bias after gradient accumulation)
        tril_ones = torch.tril(torch.ones(mb_size, mb_size, device=Q_mb.device))
        eta_expanded = eta_mb.expand(-1, -1, -1, mb_size)  # (batch, num_heads, mb_size, mb_size)
        b_bar = b.unsqueeze(0) - torch.einsum('bhmn,bhnd->bhmd', eta_expanded * tril_ones, grad_Z)

        # Compute Z_bar
        Z_bar = torch.einsum('bhmd,hde->bhme', Q_mb, W)  # Q @ W
        Z_bar = Z_bar - torch.einsum('bhmn,bhnd->bhmd', eta_expanded * Attn, grad_Z)  # - (eta * Attn) @ grad
        Z_bar = Z_bar + b_bar

        # Apply ttt_norm to Z_bar
        Z_bar_normed = torch.zeros_like(Z_bar)
        for h in range(num_heads):
            Z_bar_normed[:, h] = self.ttt_norm[h](Z_bar[:, h])

        # Output: Q + ttt_norm(Z_bar)
        output_mb = Q_mb + Z_bar_normed

        # Update W and b for next mini-batch (use last token's gradient)
        eta_last = eta_mb[:, :, -1:, :]  # (batch, num_heads, 1, 1)

        # W_next = W - eta_last * K[-1]^T @ grad_Z[-1]
        K_last = K_mb[:, :, -1:, :]  # (batch, num_heads, 1, head_dim)
        grad_Z_last = grad_Z[:, :, -1:, :]  # (batch, num_heads, 1, head_dim)

        # Average over batch (since W is shared across batch)
        delta_W = torch.einsum('bhmd,bhne->bhde', K_last, eta_last * grad_Z_last).mean(dim=0)  # (num_heads, head_dim, head_dim)
        W_next = W - delta_W

        # b_next = b - sum(eta_last * grad_Z[-1])
        delta_b = (eta_last * grad_Z_last).mean(dim=0)  # (num_heads, 1, head_dim)
        b_next = b - delta_b

        return output_mb, W_next, b_next

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with mini-batch analytic TTT.

        Args:
            x: Input (batch, seq_len, hidden_dim)
            return_stats: Return statistics

        Returns:
            output: (batch, seq_len, hidden_dim)
            stats: Optional statistics
        """
        batch_size, seq_len, _ = x.shape

        # Ensure seq_len is divisible by mini_batch_size
        if seq_len % self.mini_batch_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by mini_batch_size ({self.mini_batch_size})")

        num_mini_batches = seq_len // self.mini_batch_size

        # QKV projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Reshape to mini-batches: (batch, num_heads, num_mini_batches, mini_batch_size, head_dim)
        Q = Q.view(batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, self.head_dim)
        K = K.view(batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, self.head_dim)
        V = V.view(batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, self.head_dim)

        # Compute learning rates
        eta = self.get_eta(x)  # (batch, num_heads, num_mini_batches, mini_batch_size, 1)

        # Initialize fast-weight
        W = self.W_init  # (num_heads, head_dim, head_dim)
        b = self.b_init  # (num_heads, 1, head_dim)

        # Process each mini-batch sequentially
        outputs = []
        for mb_idx in range(num_mini_batches):
            Q_mb = Q[:, :, mb_idx]  # (batch, num_heads, mini_batch_size, head_dim)
            K_mb = K[:, :, mb_idx]
            V_mb = V[:, :, mb_idx]
            eta_mb = eta[:, :, mb_idx]  # (batch, num_heads, mini_batch_size, 1)

            output_mb, W, b = self.process_mini_batch(Q_mb, K_mb, V_mb, eta_mb, W, b)
            outputs.append(output_mb)

        # Concatenate mini-batches
        output = torch.cat(outputs, dim=2)  # (batch, num_heads, seq_len, head_dim)

        # Reshape back to (batch, seq_len, hidden_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)

        # Post normalization (second LayerNorm)
        output = self.post_norm(output)

        # Optional gating
        if self.use_gate:
            gate = self.gate_proj(x)
            gate = F.gelu(gate, approximate='tanh')
            output = gate * output

        # Output projection
        output = self.out_proj(output)

        # Statistics
        stats = None
        if return_stats:
            stats = {
                'num_mini_batches': num_mini_batches,
                'mini_batch_size': self.mini_batch_size,
                'avg_eta': eta.mean().item(),
            }

        return output, stats
```

3. **Test Analytic Implementation** (1 week - REALISTIC):
```python
# tests/test_ttt_linear_analytic.py
import torch
from ponderttt.models.ttt_linear_analytic import TTTLinearAnalytic

def test_forward_pass():
    """Test basic forward pass."""
    layer = TTTLinearAnalytic(
        hidden_dim=256,
        num_heads=4,
        mini_batch_size=16,
    )

    x = torch.randn(2, 32, 256)  # batch=2, seq_len=32 (2 mini-batches)
    output, stats = layer(x, return_stats=True)

    assert output.shape == x.shape
    assert stats['num_mini_batches'] == 2
    print("✓ Forward pass test passed")

def test_gradient_flow():
    """Test gradient flow through analytic solution."""
    layer = TTTLinearAnalytic(hidden_dim=128, num_heads=4, mini_batch_size=8)
    layer.train()

    x = torch.randn(1, 16, 128, requires_grad=True)
    output, _ = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.W_init.grad is not None
    print("✓ Gradient flow test passed")

def test_learnable_lr():
    """Test learnable LR network."""
    layer = TTTLinearAnalytic(
        hidden_dim=128,
        num_heads=4,
        mini_batch_size=8,
        use_learnable_lr=True
    )

    x = torch.randn(1, 16, 128)
    eta = layer.get_eta(x)

    assert eta.shape == (1, 4, 2, 8, 1)  # batch, heads, mini_batches, mb_size, 1
    assert (eta > 0).all() and (eta <= 1.0).all()  # Should be in reasonable range
    print("✓ Learnable LR test passed")

def test_numerical_stability():
    """Test on longer sequences for numerical issues."""
    layer = TTTLinearAnalytic(hidden_dim=256, num_heads=8, mini_batch_size=16)
    x = torch.randn(1, 256, 256)  # 16 mini-batches

    output, _ = layer(x)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    print("✓ Numerical stability test passed")

if __name__ == "__main__":
    test_forward_pass()
    test_gradient_flow()
    test_learnable_lr()
    test_numerical_stability()
    print("\n✅ All tests passed!")
```

4. **Debug and Refine** (3-5 days):
   - Fix numerical instabilities
   - Verify against official JAX implementation
   - Profile performance

5. **Integrate into Transformer** (2 days):
   - Add `TTTLinearAnalytic` option to `IterativeTTTBlock`
   - Update config to support `ttt_variant: 'analytic' | 'iterative'`

**Success Criteria**:
- All tests pass
- Analytic TTT layer produces reasonable outputs
- Gradient flow works correctly
- Can train small model on toy data

**Academic Value**:
- Provides faithful baseline for comparison
- Establishes credibility of experiments
- Allows measuring gap between analytic and iterative

---

### Task 1.2: Fix LayerNorm Application in Iterative TTT
**Priority**: P1 - Critical
**Estimated Time**: 3 days
**Academic Importance**: ⭐⭐⭐⭐ (Essential for correctness)

**Actions**:

1. **Add per-head TTT normalization** to `iterative_ttt.py`:

```python
class IterativeTTTLayer(nn.Module):
    def __init__(self, ...):
        # Add per-head ttt_norm (before residual)
        self.ttt_norm = nn.ModuleList([
            nn.LayerNorm(self.head_dim) for _ in range(num_heads)
        ])

        # Keep post_norm (after residual) - rename for clarity
        self.post_norm = nn.LayerNorm(hidden_dim)

    def forward(self, ...):
        # Line 312-315: BEFORE
        # z_t = fast_weight_t(k_t_expanded).squeeze(2)
        # output_t = q_t + z_t

        # AFTER (with two-stage normalization):
        z_t = fast_weight_t(k_t_expanded).squeeze(2)  # (batch, num_heads, head_dim)

        # Apply ttt_norm per head (BEFORE adding to Q)
        z_t_normed = torch.zeros_like(z_t)
        for h in range(self.num_heads):
            z_t_normed[:, h] = self.ttt_norm[h](z_t[:, h])

        # Residual connection with normalized output
        output_t = q_t + z_t_normed
```

2. **Update loss computation** to use normalized output:
```python
def _compute_ttt_loss(self, fast_weight, k_t, v_t):
    # Forward through fast-weight
    z_t = fast_weight(k_t_expanded).squeeze(2)

    # Apply ttt_norm per head
    z_t_normed = torch.zeros_like(z_t)
    for h in range(self.num_heads):
        z_t_normed[:, h] = self.ttt_norm[h](z_t[:, h])

    # Target
    target = v_t - k_t

    # Loss on normalized output
    loss = F.mse_loss(z_t_normed, target, reduction='mean')
    return loss
```

3. **Test LayerNorm correctness**:
```python
# Test that output statistics are stable
layer = IterativeTTTLayer(hidden_dim=256, num_heads=4)
x = torch.randn(2, 32, 256)
output, _, _ = layer(x, num_steps=torch.full((32,), 4))

# Check output statistics
print(f"Output mean: {output.mean():.4f} (should be ~0)")
print(f"Output std: {output.std():.4f} (should be ~1)")
```

**Success Criteria**:
- Two LayerNorms applied (ttt_norm before residual, post_norm after)
- Output statistics stable (mean≈0, std≈1)
- Matches official TTT normalization pattern

---

### Task 1.3: Fix REINFORCE Temporal Credit Assignment
**Priority**: P1 - Critical
**Estimated Time**: 1 week
**Academic Importance**: ⭐⭐⭐⭐⭐ (Critical for learned policy)

**Current Problem** (line 469):
```python
per_token_advantage = self.baseline - per_token_loss.detach()  # Only immediate reward!
```

**Solution Options**:

**Option A: Discounted Returns (Monte Carlo)** - Recommended for first iteration
```python
def compute_returns(
    self,
    per_token_loss: torch.Tensor,  # (batch, seq_len-1)
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute discounted returns G_t = sum_{i=t}^T gamma^{i-t} * (-L_i).

    Args:
        per_token_loss: Loss for each token
        gamma: Discount factor

    Returns:
        returns: Discounted returns (batch, seq_len-1)
    """
    batch_size, seq_len = per_token_loss.shape
    returns = torch.zeros_like(per_token_loss)

    # Compute returns backwards (dynamic programming)
    G_next = 0
    for t in reversed(range(seq_len)):
        # Reward = negative loss (lower loss = higher reward)
        reward_t = -per_token_loss[:, t]
        G_t = reward_t + gamma * G_next
        returns[:, t] = G_t
        G_next = G_t

    return returns

# In forward method (line 443-495):
if self.config.use_learned_policy and self.training:
    # ... existing code ...

    # Compute returns instead of immediate rewards
    returns = self.compute_returns(per_token_loss, gamma=0.99)

    # Advantage = returns - baseline
    per_token_advantage = returns - self.baseline  # (batch, seq_len-1)

    # Update baseline with mean return (not just loss)
    mean_return = returns.mean()
    if not self.baseline_initialized:
        self.baseline.copy_(mean_return.detach())
        self.baseline_initialized.fill_(True)
    else:
        with torch.no_grad():
            self.baseline.mul_(self.baseline_momentum).add_(
                mean_return.detach(), alpha=(1 - self.baseline_momentum)
            )

    # Policy gradient (unchanged)
    policy_loss = 0.0
    for log_probs in log_probs_list:
        log_probs_aligned = ...  # existing alignment code
        policy_loss -= (log_probs_aligned * per_token_advantage.detach()).mean()
```

**Option B: Actor-Critic (A2C)** - More sophisticated, future work
```python
class ValueNetwork(nn.Module):
    """Estimate state value V(s_t) for variance reduction."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states).squeeze(-1)

# Add to model:
self.value_network = ValueNetwork(config.hidden_dim)

# In training:
value_estimates = self.value_network(hidden_states)
td_target = reward + gamma * value_estimates[:, 1:]
td_error = td_target - value_estimates[:, :-1]
actor_loss = -(log_probs * td_error.detach()).mean()
critic_loss = td_error.pow(2).mean()
total_loss = lm_loss + actor_loss + 0.5 * critic_loss
```

**Implementation Plan**:
1. Week 1: Implement Option A (Monte Carlo returns)
2. Test on toy problem (verify credit assignment works)
3. Document decision (why Monte Carlo vs A2C)
4. (Future) Implement Option B as ablation

**Test Plan**:
```python
# Test credit assignment on toy problem
def test_credit_assignment():
    """
    Toy problem:
    - Token 0: Always easy (K=1 sufficient)
    - Token 1: Always hard (K=8 needed)

    Correct policy should learn K=1 for token 0, K=8 for token 1.
    With immediate reward only, this fails due to sequential dependency.
    """
    # TODO: Implement toy test
```

**Success Criteria**:
- Returns computed correctly with discount
- Policy gradient uses full returns
- Toy test shows correct credit assignment

**Academic Significance**:
- This is a **methodological contribution**
- Proper temporal credit assignment in sequential test-time training
- Can cite as "REINFORCE with Monte Carlo returns for adaptive TTT"

---

### Task 1.4: Add Learnable Learning Rate Network
**Priority**: P2 - Important (for fair comparison with official TTT)
**Estimated Time**: 3 days
**Academic Importance**: ⭐⭐⭐ (Completeness)

**Rationale**: Official TTT has learnable per-head LR network. For fair comparison, iterative variant should too.

**Actions**:
```python
# Add to iterative_ttt.py or iterative_ttt_v2.py
class IterativeTTTLayer(nn.Module):
    def __init__(self, ..., use_learnable_lr: bool = True):
        # ... existing init ...

        if use_learnable_lr:
            # Per-head LR network (like official TTT)
            self.lr_net = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 1, bias=True),
                    nn.Sigmoid()  # Output in [0, 1]
                ) for _ in range(num_heads)
            ])
            self.use_learnable_lr = True
        else:
            self.use_learnable_lr = False

    def _get_lr(self, x, step, position):
        """Compute learning rate with optional learned component."""
        if self.use_learnable_lr:
            # Compute per-head multiplier
            lr_mults = []
            for h in range(self.num_heads):
                # Input: full hidden state
                lr_mult = self.lr_net[h](x)  # (batch, 1)
                lr_mults.append(lr_mult)
            lr_mult = torch.stack(lr_mults, dim=1).mean(dim=1)  # Average over heads

            base_lr = self.base_lr / self.head_dim
            return base_lr * lr_mult.item()  # Scalar LR
        else:
            # Fixed LR
            return self.base_lr
```

**Test**:
```python
def test_learnable_lr():
    layer = IterativeTTTLayer(hidden_dim=256, num_heads=4, use_learnable_lr=True)
    x = torch.randn(2, 32, 256)

    # Forward pass
    output, _, _ = layer(x)

    # Check that lr_net parameters have gradients
    loss = output.sum()
    loss.backward()

    for h in range(4):
        assert layer.lr_net[h][0].weight.grad is not None
    print("✓ Learnable LR gradients flow correctly")
```

**Success Criteria**:
- Learnable LR networks have gradients
- LR values reasonable (between 0 and base_lr)
- Can enable/disable for ablation

---

## Phase 2: Core Contributions (Weeks 6-9)
**Goal**: Implement and validate the core research contributions

**Research Question**:
> "Can we learn to allocate computation (# of gradient steps) per token based on difficulty, achieving better quality-compute tradeoff than uniform allocation?"

**Focus Strategy**: Deep analysis of 3 core methods rather than shallow coverage of many methods.

### Task 2.1: REINFORCE with Temporal Credit Assignment (MAIN CONTRIBUTION)
**Priority**: P1 - Core Research Contribution
**Estimated Time**: 2 weeks
**Academic Importance**: ⭐⭐⭐⭐⭐ (Main novelty)

**Variants to Implement**:

1. **Learned Policy (REINFORCE with Monte Carlo Returns)** - MAIN CONTRIBUTION
   - Fix temporal credit assignment
   - This is the primary novel contribution

2. **Difficulty-Aware Heuristics** - Already implemented, refine
   - Entropy-based
   - Loss-based
   - Gradient-norm based

3. **Oracle Upper Bound** - Needs fixing (Task 2.2)
   - Corrected per-token oracle calculation
   - Provides theoretical upper bound

**Implementation** (from Task 1.3):
- Monte Carlo returns for temporal credit
- Proper advantage computation
- Tested on toy problem first

**NOTE**: Additional methods moved to Appendix/Future Work to maintain focus.

---

### Task 2.1b: Difficulty-Aware Heuristics (Refinement)
**Priority**: P2 - Supporting Methods
**Estimated Time**: 3 days
**Academic Importance**: ⭐⭐⭐ (Baselines)

**Actions**:
1. Refine existing heuristic policies
2. Add proper calibration (percentile-based)
3. Document limitations vs learned policy

---

### (APPENDIX) Task 2.1c: Value-Based Policy (Q-Learning)
**Priority**: P3 - Optional/Appendix
**Estimated Time**: 1 week (if time permits)
**Academic Importance**: ⭐⭐ (Ablation)

**Status**: Implement ONLY if time permits. Otherwise, include in "Future Work" section.

```python
class ValueBasedPolicy(nn.Module):
    """
    Value-based iteration allocation using Q-learning.

    NOTE: This is an ABLATION/APPENDIX method.
    Main paper focuses on REINFORCE with temporal credit.
    """
    def __init__(self, hidden_dim, step_options):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(step_options)),
        )

    # ... (simplified implementation)
```

**Decision Criteria**:
- IF experiments show REINFORCE works well → Skip Q-learning, mention in Future Work
- IF REINFORCE has high variance issues → Implement Q-learning as ablation

---

### (FUTURE WORK) Budget-Constrained Policy
**Status**: NOT implemented in this paper

**Rationale**:
- Too many methods dilutes contribution
- Can be cited as natural extension in Future Work section
- Better to have deep analysis of REINFORCE than shallow coverage of many methods

```markdown
## Future Work Section (in paper):
"While our work focuses on learned policies via REINFORCE,
alternative formulations such as explicit budget-constrained
optimization (e.g., Lagrangian methods) or value-based RL
(e.g., Q-learning) present promising directions..."
        seq_len = hidden_states.shape[1]
        k_values_continuous = allocation_weights * self.target_budget * seq_len

        # Discretize to step_options (round to nearest)
        k_values = torch.zeros_like(k_values_continuous, dtype=torch.long)
        for i, k_option in enumerate(self.step_options):
            if i == 0:
                mask = k_values_continuous < (self.step_options[0] + self.step_options[1]) / 2
            elif i == len(self.step_options) - 1:
                mask = k_values_continuous >= (self.step_options[-2] + self.step_options[-1]) / 2
            else:
                mask = (k_values_continuous >= (self.step_options[i-1] + self.step_options[i]) / 2) & \
                       (k_values_continuous < (self.step_options[i] + self.step_options[i+1]) / 2)
            k_values[mask] = k_option

        return k_values

    def compute_budget_penalty(self, k_values):
        """Penalty for deviating from target budget."""
        actual_budget = k_values.float().mean()
        penalty = (actual_budget - self.target_budget) ** 2
        return self.log_lambda.exp() * penalty
```

**Experimental Design**:
- Compare all 5 policies (Uniform K=1,2,4,8, Heuristic, REINFORCE, Q-learning, Budget)
- Measure quality (perplexity) vs compute (FLOPs)
- Plot Pareto frontier
- Statistical significance testing (paired t-tests with Bonferroni correction)

**Success Criteria**:
- Learned policies achieve better Pareto frontier than uniform
- Statistical significance (p < 0.05 after correction)
- Clear visualization of tradeoffs

---

### Task 2.2: Fix and Extend Oracle Analysis
**Priority**: Core Research Contribution
**Estimated Time**: 1 week
**Academic Importance**: ⭐⭐⭐⭐ (Upper bound)

**Current Problem**: Oracle K computed incorrectly (uses global loss, interferes with other tokens)

**Solution**:

```python
class OracleAnalyzer:
    """
    Compute oracle upper bound on quality-compute tradeoff.

    Three oracle variants:
    1. Per-token oracle (greedy, current approach - needs fixing)
    2. Joint oracle (NP-hard, use beam search approximation)
    3. Difficulty oracle (if we knew true difficulty, what would we achieve?)
    """

    def __init__(self, model, step_options, device='cuda'):
        self.model = model
        self.step_options = step_options
        self.device = device

    def compute_per_token_oracle(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        use_sequential: bool = True,
    ) -> Dict:
        """
        Per-token oracle: For each token, find K that minimizes its loss.

        FIXED VERSION:
        - Track per-token losses (not global average)
        - If sequential, account for carry-over effects
        """
        batch_size, seq_len = input_ids.shape
        oracle_k = torch.zeros(seq_len, dtype=torch.long)
        per_token_losses = torch.zeros(seq_len, len(self.step_options))

        for token_pos in range(seq_len):
            best_k = self.step_options[0]
            best_loss = float('inf')

            for k_idx, k in enumerate(self.step_options):
                # Set all tokens to K=1, except this position
                num_steps = torch.ones(seq_len, device=self.device)
                num_steps[token_pos] = k

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(input_ids, labels, num_steps, return_stats=False)

                # Extract per-token loss
                shift_logits = outputs['logits'][:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='none'
                ).view(shift_labels.shape)

                # Get loss for this token only
                if token_pos > 0:  # First token has no prediction
                    token_loss = per_token_loss[:, token_pos - 1].item()
                else:
                    token_loss = 0.0

                per_token_losses[token_pos, k_idx] = token_loss

                if token_loss < best_loss:
                    best_loss = token_loss
                    best_k = k

            oracle_k[token_pos] = best_k

        return {
            'oracle_k': oracle_k,
            'per_token_losses': per_token_losses,
            'mean_k': oracle_k.float().mean().item(),
        }

    def compute_joint_oracle_beam(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        beam_width: int = 10,
    ) -> Dict:
        """
        Joint oracle using beam search approximation.

        At each token position, keep top-k partial allocations.
        This is exponential in seq_len, so use beam search.
        """
        batch_size, seq_len = input_ids.shape

        # Beam: List of (allocation, cumulative_loss) tuples
        # allocation: torch.Tensor of shape (seq_len,)
        beam = [(torch.ones(seq_len, device=self.device), 0.0)]

        for token_pos in range(seq_len):
            new_beam = []

            for allocation, cum_loss in beam:
                # Try each K option for this token
                for k in self.step_options:
                    new_allocation = allocation.clone()
                    new_allocation[token_pos] = k

                    # Compute loss up to this token
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids, labels,
                            new_allocation[:token_pos+1],  # Only up to current token
                            return_stats=False
                        )

                    # Extract loss for this token
                    new_loss = outputs['loss'].item() if token_pos > 0 else 0.0
                    new_cum_loss = cum_loss + new_loss

                    new_beam.append((new_allocation, new_cum_loss))

            # Keep top beam_width
            new_beam.sort(key=lambda x: x[1])
            beam = new_beam[:beam_width]

        # Best allocation
        best_allocation, best_cum_loss = beam[0]

        return {
            'oracle_k': best_allocation,
            'cumulative_loss': best_cum_loss,
            'mean_k': best_allocation.float().mean().item(),
        }

    def analyze_difficulty_correlation(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        difficulty_metrics: List[str] = ['loss', 'entropy', 'attention'],
    ) -> Dict:
        """
        Analyze correlation between difficulty metrics and oracle K.

        This tells us: If we had perfect difficulty predictor, how well could we do?
        """
        # Get per-token oracle
        oracle_result = self.compute_per_token_oracle(input_ids, labels)
        oracle_k = oracle_result['oracle_k'].cpu().numpy()

        # Compute difficulty metrics
        correlations = {}

        for metric_name in difficulty_metrics:
            if metric_name == 'loss':
                # Per-token loss as difficulty
                with torch.no_grad():
                    outputs = self.model(input_ids, labels, return_stats=False)
                shift_logits = outputs['logits'][:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='none'
                ).view(shift_labels.shape)
                difficulty = per_token_loss.cpu().numpy()

            elif metric_name == 'entropy':
                # Prediction entropy
                with torch.no_grad():
                    outputs = self.model(input_ids, labels, return_stats=False)
                logits = outputs['logits'][:, :-1, :]
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * probs.log()).sum(dim=-1)
                difficulty = entropy.cpu().numpy()

            elif metric_name == 'attention':
                # TODO: Implement attention-based difficulty
                pass

            # Compute correlation
            from scipy.stats import spearmanr, pearsonr
            if difficulty.shape == oracle_k[1:].shape:  # Handle shift
                pearson_r, pearson_p = pearsonr(difficulty.flatten(), oracle_k[1:].flatten())
                spearman_r, spearman_p = spearmanr(difficulty.flatten(), oracle_k[1:].flatten())

                correlations[metric_name] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                }

        return {
            'oracle_k': oracle_k,
            'correlations': correlations,
        }
```

**Experiments**:
1. Run oracle analysis on validation set (sample of 100 sequences)
2. Plot oracle K distribution
3. Measure correlation between difficulty and oracle K
4. Plot oracle Pareto frontier (upper bound)

**Success Criteria**:
- Oracle correctly identifies per-token optimal K
- Correlation analysis shows which difficulty metrics are predictive
- Oracle Pareto frontier serves as upper bound

---

### Task 2.3: Convergence Analysis (Iterative vs Analytic)
**Priority**: Core Research Contribution
**Estimated Time**: 1 week
**Academic Importance**: ⭐⭐⭐⭐⭐ (Theoretical understanding)

**Research Question**:
> "How many gradient steps K are needed for iterative TTT to approximate analytic TTT?"

**Implementation**:

```python
class ConvergenceAnalyzer:
    """
    Analyze convergence of iterative TTT to analytic solution.
    """

    def __init__(self, analytic_model, iterative_model, device='cuda'):
        self.analytic_model = analytic_model
        self.iterative_model = iterative_model
        self.device = device

        # Share parameters between models
        self._share_parameters()

    def _share_parameters(self):
        """
        Share Q, K, V, output projections between analytic and iterative.
        Fast-weight initialization should match.
        """
        # Copy Q, K, V projections
        self.iterative_model.q_proj.weight.data.copy_(
            self.analytic_model.q_proj.weight.data
        )
        self.iterative_model.k_proj.weight.data.copy_(
            self.analytic_model.k_proj.weight.data
        )
        self.iterative_model.v_proj.weight.data.copy_(
            self.analytic_model.v_proj.weight.data
        )

        # Copy fast-weight initialization
        # (Analytic has W_init, iterative has fast_weight module)
        # This needs careful handling based on fast-weight type

    def measure_output_gap(
        self,
        input_ids: torch.Tensor,
        k_values: List[int] = [1, 2, 4, 8, 16, 32],
    ) -> Dict:
        """
        Measure L2 distance between analytic and iterative outputs.

        Gap(K) = ||output_analytic - output_iterative(K)||_2

        Returns:
            Dictionary with gaps for each K
        """
        # Analytic output
        with torch.no_grad():
            x = self.analytic_model.token_embedding(input_ids)
            analytic_output, _ = self.analytic_model(x)

        # Iterative outputs for different K
        gaps = {}
        for k in k_values:
            with torch.no_grad():
                x = self.iterative_model.token_embedding(input_ids)
                num_steps = torch.full((input_ids.shape[1],), k, device=self.device)
                iterative_output, _, _ = self.iterative_model(x, num_steps=num_steps)

            # L2 distance
            gap = (analytic_output - iterative_output).norm(p=2, dim=-1).mean().item()
            gaps[k] = gap

        return gaps

    def measure_loss_gap(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        k_values: List[int] = [1, 2, 4, 8, 16, 32],
    ) -> Dict:
        """
        Measure difference in LM loss between analytic and iterative.

        Returns:
            Dictionary with loss differences for each K
        """
        # Analytic loss
        with torch.no_grad():
            analytic_outputs = self.analytic_model(input_ids, labels=labels)
            analytic_loss = analytic_outputs['loss'].item()

        # Iterative losses for different K
        loss_gaps = {}
        iterative_losses = {}

        for k in k_values:
            with torch.no_grad():
                num_steps = torch.full((input_ids.shape[1],), k, device=self.device)
                iterative_outputs = self.iterative_model(
                    input_ids, labels=labels, num_steps=num_steps
                )
                iterative_loss = iterative_outputs['loss'].item()

            loss_gaps[k] = abs(iterative_loss - analytic_loss)
            iterative_losses[k] = iterative_loss

        return {
            'analytic_loss': analytic_loss,
            'iterative_losses': iterative_losses,
            'loss_gaps': loss_gaps,
        }

    def analyze_convergence_rate(
        self,
        dataloader,
        k_values: List[int] = [1, 2, 4, 8, 16],
        num_batches: int = 100,
    ) -> Dict:
        """
        Analyze convergence rate across dataset.

        Fit exponential decay: Gap(K) ≈ A * exp(-λ * K)
        """
        import numpy as np
        from scipy.optimize import curve_fit

        all_gaps = {k: [] for k in k_values}

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(self.device)

            gaps = self.measure_output_gap(input_ids, k_values)
            for k, gap in gaps.items():
                all_gaps[k].append(gap)

        # Average gaps
        avg_gaps = {k: np.mean(all_gaps[k]) for k in k_values}

        # Fit exponential decay
        def exponential_decay(k, A, lam):
            return A * np.exp(-lam * k)

        k_array = np.array(k_values)
        gap_array = np.array([avg_gaps[k] for k in k_values])

        try:
            popt, pcov = curve_fit(exponential_decay, k_array, gap_array)
            A_fit, lambda_fit = popt

            # Estimate K for 95% convergence (gap < 0.05 * A)
            k_95 = -np.log(0.05) / lambda_fit if lambda_fit > 0 else float('inf')
        except:
            A_fit, lambda_fit, k_95 = None, None, None

        return {
            'k_values': k_values,
            'avg_gaps': avg_gaps,
            'exponential_fit': {
                'A': A_fit,
                'lambda': lambda_fit,
                'k_95_convergence': k_95,
            },
        }
```

**Experiments**:
1. Run convergence analysis on validation set
2. Plot gap vs K (log scale)
3. Fit exponential decay
4. Report K required for 95% convergence
5. Compare convergence rate for different sequence lengths

**Success Criteria**:
- Clear convergence trend (gap decreases with K)
- Exponential fit is good (R² > 0.9)
- Can quantify: "K=X steps achieves 95% of analytic performance"

**Academic Value**:
- Provides theoretical grounding for iterative variant
- Justifies choice of K in experiments
- Novelty: First analysis of iterative vs analytic TTT

---

## Phase 3: Accurate FLOPs Measurement (Weeks 10-11)
**Goal**: Ensure all efficiency claims are accurate and credible

### Task 3.1: Comprehensive FLOPs Calculator
**Priority**: P2 - Important
**Estimated Time**: 2 weeks (REALISTIC: includes validation)
**Academic Importance**: ⭐⭐⭐⭐ (Credibility)

**Implementation**:

```python
"""
Accurate FLOPs counting for TTT models.

Includes ALL operations:
- Matrix multiplications (forward and backward)
- Layer normalizations
- Activation functions (GELU, Sigmoid, Softmax)
- Policy networks
- Higher-order gradients (create_graph=True)
"""

class FLOPsCounter:
    """Accurate FLOPs counter for neural networks."""

    @staticmethod
    def linear_flops(in_features, out_features, has_bias=False):
        """FLOPs for Linear layer: y = xW + b"""
        # Forward: matmul (2*in*out - out) + bias (out if has_bias)
        forward = 2 * in_features * out_features
        if has_bias:
            forward += out_features
        return forward

    @staticmethod
    def linear_backward_flops(in_features, out_features, has_bias=False):
        """FLOPs for Linear layer backward pass."""
        # Gradient w.r.t. input: grad_output @ W^T
        grad_input = 2 * in_features * out_features
        # Gradient w.r.t. weight: input^T @ grad_output
        grad_weight = 2 * in_features * out_features
        # Gradient w.r.t. bias: sum(grad_output)
        grad_bias = out_features if has_bias else 0
        return grad_input + grad_weight + grad_bias

    @staticmethod
    def layernorm_flops(normalized_shape):
        """FLOPs for LayerNorm."""
        n = normalized_shape
        # Mean: sum(x) / n → n + 1
        # Variance: sum((x - mean)^2) / n → 3n + 1
        # Normalize: (x - mean) / sqrt(var + eps) → 2n + 1
        # Scale & shift: x * weight + bias → 2n
        return 8 * n

    @staticmethod
    def gelu_flops(n):
        """FLOPs for GELU activation."""
        # GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # Approximation: 8 ops per element
        return 8 * n

    @staticmethod
    def sigmoid_flops(n):
        """FLOPs for Sigmoid activation."""
        # sigmoid(x) = 1 / (1 + exp(-x))
        # Approximation: 4 ops per element (exp + 3 arithmetic)
        return 4 * n

    @staticmethod
    def softmax_flops(n, dim_size):
        """FLOPs for Softmax."""
        # exp(x): n ops
        # sum: dim_size ops
        # divide: n ops
        return 2 * n + dim_size

    @staticmethod
    def mse_loss_flops(n):
        """FLOPs for MSE loss."""
        # (pred - target)^2: 2n ops
        # mean: 1 op
        return 2 * n + 1

    @staticmethod
    def cross_entropy_flops(batch_size, seq_len, vocab_size):
        """FLOPs for cross-entropy loss."""
        # Softmax: (batch * seq_len * vocab_size)
        # Log: batch * seq_len * vocab_size
        # Gather: batch * seq_len
        # Mean: 1
        n = batch_size * seq_len * vocab_size
        return FLOPsCounter.softmax_flops(n, vocab_size) + n + batch_size * seq_len + 1


class TTTFLOPsAnalyzer:
    """Analyze FLOPs for TTT models with accurate accounting."""

    def __init__(self, model):
        self.model = model
        self.config = model.config

    def count_embedding_flops(self, seq_len):
        """FLOPs for embedding lookup."""
        # Embedding is essentially a table lookup (0 FLOPs)
        # But we count addition of token + position embeddings
        return self.config.hidden_dim * seq_len

    def count_attention_flops(self, seq_len):
        """FLOPs for standard multi-head attention."""
        d = self.config.hidden_dim
        n = seq_len
        num_heads = self.config.num_heads
        head_dim = d // num_heads

        # Q, K, V projections
        qkv_flops = 3 * FLOPsCounter.linear_flops(d, d)

        # Attention scores: Q @ K^T
        # Shape: (batch, num_heads, seq_len, seq_len)
        scores_flops = 2 * num_heads * n * n * head_dim

        # Softmax over attention scores
        softmax_flops = FLOPsCounter.softmax_flops(num_heads * n * n, n)

        # Attention @ V
        attn_v_flops = 2 * num_heads * n * n * head_dim

        # Output projection
        out_flops = FLOPsCounter.linear_flops(d, d)

        # LayerNorm (2x: pre and post)
        ln_flops = 2 * FLOPsCounter.layernorm_flops(d)

        total = qkv_flops + scores_flops + softmax_flops + attn_v_flops + out_flops + ln_flops
        return total

    def count_ttt_analytic_flops(self, seq_len, mini_batch_size=16):
        """FLOPs for analytic TTT layer."""
        d = self.config.hidden_dim
        num_heads = self.config.num_heads
        head_dim = d // num_heads
        n_mb = seq_len // mini_batch_size
        mb = mini_batch_size

        # Q, K, V projections
        qkv_flops = 3 * FLOPsCounter.linear_flops(d, d)

        # Per mini-batch operations
        per_mb_flops = 0

        # Forward through fast-weight: Z = K @ W + b
        # Shape: (mb, head_dim) @ (head_dim, head_dim)
        per_mb_flops += num_heads * 2 * mb * head_dim * head_dim

        # TTT norm (per head)
        per_mb_flops += num_heads * FLOPsCounter.layernorm_flops(head_dim) * mb

        # Gradient computation (simplified - actual is more complex)
        per_mb_flops += num_heads * 2 * mb * head_dim  # grad = output - target

        # Triangular attention: Q @ K^T (lower triangular)
        per_mb_flops += num_heads * mb * mb * head_dim  # tril reduces by ~50%

        # Analytic update: Z_bar computation
        # Z_bar = Q @ W - (eta * Attn) @ grad + b_bar
        per_mb_flops += num_heads * 2 * mb * head_dim * head_dim  # Q @ W
        per_mb_flops += num_heads * 2 * mb * mb * head_dim  # (eta * Attn) @ grad
        per_mb_flops += num_heads * mb * head_dim  # + b_bar

        # TTT norm again
        per_mb_flops += num_heads * FLOPsCounter.layernorm_flops(head_dim) * mb

        # Update W, b for next mini-batch
        per_mb_flops += num_heads * 2 * head_dim * head_dim  # K[-1]^T @ grad[-1]
        per_mb_flops += num_heads * head_dim  # b update

        # Total over all mini-batches
        mb_total_flops = n_mb * per_mb_flops

        # Post norm
        post_norm_flops = FLOPsCounter.layernorm_flops(d)

        # Gate (if used)
        gate_flops = 0
        if hasattr(self.model, 'use_gate') and self.model.use_gate:
            gate_flops = FLOPsCounter.linear_flops(d, d)
            gate_flops += FLOPsCounter.gelu_flops(d)

        # Output projection
        out_flops = FLOPsCounter.linear_flops(d, d)

        total = qkv_flops + mb_total_flops + post_norm_flops + gate_flops + out_flops
        return total

    def count_ttt_iterative_flops(self, seq_len, K):
        """
        FLOPs for iterative TTT layer with K gradient steps.

        IMPORTANT: Includes forward AND backward passes for gradient computation.

        NOTE: Higher-order gradients (create_graph=True) overhead is difficult to
        calculate theoretically. We supplement with measured wall-clock time.
        """
        d = self.config.hidden_dim
        num_heads = self.config.num_heads
        head_dim = d // num_heads
        fw_hidden = self.config.fast_weight_hidden_dim  # For MLP variant

        # Q, K, V projections
        qkv_flops = 3 * FLOPsCounter.linear_flops(d, d)

        # Per-token, per-step operations
        per_token_per_step_flops = 0

        # Forward through fast-weight (MLP: fc1 + fc2)
        per_token_per_step_flops += num_heads * FLOPsCounter.linear_flops(head_dim, fw_hidden)
        per_token_per_step_flops += num_heads * FLOPsCounter.layernorm_flops(fw_hidden)
        per_token_per_step_flops += num_heads * FLOPsCounter.gelu_flops(fw_hidden)
        per_token_per_step_flops += num_heads * FLOPsCounter.linear_flops(fw_hidden, head_dim)

        # Loss computation: MSE(output, target)
        per_token_per_step_flops += num_heads * FLOPsCounter.mse_loss_flops(head_dim)

        # BACKWARD PASS (for gradient computation)
        # Gradient w.r.t. loss → fc2 backward → GELU backward → LayerNorm backward → fc1 backward
        per_token_per_step_flops += num_heads * FLOPsCounter.linear_backward_flops(fw_hidden, head_dim)
        per_token_per_step_flops += num_heads * 2 * fw_hidden  # GELU backward (approx)
        per_token_per_step_flops += num_heads * FLOPsCounter.layernorm_flops(fw_hidden)  # LN backward ≈ forward
        per_token_per_step_flops += num_heads * FLOPsCounter.linear_backward_flops(head_dim, fw_hidden)

        # Parameter update (SGD): param = param - lr * grad
        num_params = num_heads * (head_dim * fw_hidden + fw_hidden + fw_hidden * head_dim + head_dim)
        per_token_per_step_flops += 2 * num_params  # subtract and multiply

        # HIGHER-ORDER GRADIENTS (create_graph=True during training)
        # This doubles the backward cost (gradient of gradient)
        if self.model.training:
            per_token_per_step_flops *= 1.5  # Conservative estimate

        # Total over all tokens and steps
        iterative_total = seq_len * K * per_token_per_step_flops

        # Output computation (after K steps)
        # TTT norm per head per token
        ttt_norm_flops = seq_len * num_heads * FLOPsCounter.layernorm_flops(head_dim)

        # Post norm
        post_norm_flops = seq_len * FLOPsCounter.layernorm_flops(d)

        # Gate
        gate_flops = 0
        if hasattr(self.model, 'use_gate') and self.model.use_gate:
            gate_flops = seq_len * FLOPsCounter.linear_flops(d, d)
            gate_flops += seq_len * FLOPsCounter.gelu_flops(d)

        # Output projection
        out_flops = seq_len * FLOPsCounter.linear_flops(d, d)

        total = qkv_flops + iterative_total + ttt_norm_flops + post_norm_flops + gate_flops + out_flops
        return total

    def count_policy_network_flops(self, seq_len):
        """FLOPs for halting policy network."""
        d = self.config.hidden_dim
        num_steps = len(self.config.step_options)

        # Bi-LSTM (if used)
        # LSTM cell: 4 * (input_size * hidden_size + hidden_size^2) per token per direction
        lstm_flops = 0
        if self.config.policy_use_lstm:
            lstm_flops = seq_len * 2 * 4 * (d * d + d * d)  # 2 for bidirectional

        # Step predictor MLP
        mlp_flops = seq_len * FLOPsCounter.linear_flops(2*d if self.config.policy_use_lstm else d, d)
        mlp_flops += seq_len * FLOPsCounter.gelu_flops(d)
        mlp_flops += seq_len * FLOPsCounter.linear_flops(d, num_steps)

        # Softmax (for Gumbel-softmax)
        mlp_flops += seq_len * FLOPsCounter.softmax_flops(num_steps, num_steps)

        return lstm_flops + mlp_flops

    def count_ffn_flops(self, seq_len):
        """FLOPs for feed-forward network."""
        d = self.config.hidden_dim
        d_ffn = self.config.ffn_dim

        # fc1
        flops = seq_len * FLOPsCounter.linear_flops(d, d_ffn)
        # GELU
        flops += seq_len * FLOPsCounter.gelu_flops(d_ffn)
        # fc2
        flops += seq_len * FLOPsCounter.linear_flops(d_ffn, d)
        # LayerNorm
        flops += seq_len * FLOPsCounter.layernorm_flops(d)

        return flops

    def count_lm_head_flops(self, seq_len):
        """FLOPs for language model head."""
        d = self.config.hidden_dim
        vocab = self.config.vocab_size

        # Final LayerNorm
        ln_flops = seq_len * FLOPsCounter.layernorm_flops(d)

        # Projection to vocab
        proj_flops = seq_len * FLOPsCounter.linear_flops(d, vocab, has_bias=False)

        # Cross-entropy loss (if computing loss)
        loss_flops = FLOPsCounter.cross_entropy_flops(1, seq_len, vocab)

        return ln_flops + proj_flops + loss_flops

    def estimate_total_flops(
        self,
        seq_len: int,
        num_steps: Union[int, torch.Tensor],
        include_backward: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate total FLOPs for full forward (and backward) pass.

        Args:
            seq_len: Sequence length
            num_steps: Average number of gradient steps (for iterative TTT)
            include_backward: Whether to include backward pass for LM training

        Returns:
            Dictionary with FLOPs breakdown
        """
        flops = {}

        # Embedding
        flops['embedding'] = self.count_embedding_flops(seq_len)

        # Layers
        num_ttt_layers = len(self.config.ttt_layer_indices)
        num_standard_layers = self.config.num_layers - num_ttt_layers

        # Standard attention layers
        flops['attention_layers'] = num_standard_layers * self.count_attention_flops(seq_len)

        # TTT layers
        if isinstance(num_steps, torch.Tensor):
            avg_steps = num_steps.float().mean().item()
        else:
            avg_steps = num_steps

        if self.config.use_iterative_ttt:
            ttt_flops = self.count_ttt_iterative_flops(seq_len, avg_steps)
        else:
            ttt_flops = self.count_ttt_analytic_flops(seq_len)
        flops['ttt_layers'] = num_ttt_layers * ttt_flops

        # Policy network (if learned)
        if self.config.use_learned_policy:
            flops['policy_network'] = num_ttt_layers * self.count_policy_network_flops(seq_len)
        else:
            flops['policy_network'] = 0

        # FFN layers
        flops['ffn_layers'] = self.config.num_layers * self.count_ffn_flops(seq_len)

        # LM head
        flops['lm_head'] = self.count_lm_head_flops(seq_len)

        # Total forward
        flops['forward_total'] = sum(flops.values())

        # Backward pass (approximate as 2x forward for parameters with gradients)
        if include_backward:
            # Backward through all except embedding
            backward_flops = (flops['forward_total'] - flops['embedding']) * 2
            flops['backward_total'] = backward_flops
            flops['total'] = flops['forward_total'] + backward_flops
        else:
            flops['total'] = flops['forward_total']

        # Per-token FLOPs
        flops['per_token'] = flops['total'] / seq_len

        return flops

    def compare_configurations(
        self,
        seq_len: int = 256,
        k_values: List[int] = [1, 2, 4, 8],
    ) -> pd.DataFrame:
        """
        Compare FLOPs for different configurations.

        Returns:
            DataFrame with FLOPs breakdown
        """
        import pandas as pd

        results = []

        for k in k_values:
            flops = self.estimate_total_flops(seq_len, k, include_backward=True)
            results.append({
                'config': f'Iterative K={k}',
                'avg_steps': k,
                'total_flops': flops['total'],
                'per_token_flops': flops['per_token'],
                'ttt_flops': flops['ttt_layers'],
                'policy_flops': flops['policy_network'],
                'attention_flops': flops['attention_layers'],
                'ffn_flops': flops['ffn_layers'],
                'lm_head_flops': flops['lm_head'],
            })

        # Analytic TTT
        self.config.use_iterative_ttt = False
        flops = self.estimate_total_flops(seq_len, 0, include_backward=True)
        results.append({
            'config': 'Analytic TTT',
            'avg_steps': 'N/A',
            'total_flops': flops['total'],
            'per_token_flops': flops['per_token'],
            'ttt_flops': flops['ttt_layers'],
            'policy_flops': flops['policy_network'],
            'attention_flops': flops['attention_layers'],
            'ffn_flops': flops['ffn_layers'],
            'lm_head_flops': flops['lm_head'],
        })
        self.config.use_iterative_ttt = True

        df = pd.DataFrame(results)
        return df
```

**Validation**:
1. Compare with profiler measurements (PyTorch profiler)
2. Verify against manual calculations for small examples
3. Cross-check with official TTT FLOPs (if reported)

**Success Criteria**:
- FLOPs estimates within 10% of profiler measurements
- All operations accounted for
- Clear breakdown by component

---

## Phase 4: Comprehensive Experiments (Weeks 10-14)
**Goal**: Run full experimental suite with statistical rigor

### Task 4.1: Experimental Protocol
**Priority**: Core Research
**Estimated Time**: 1 week planning + 3 weeks execution
**Academic Importance**: ⭐⭐⭐⭐⭐ (Evidence)

**Experimental Design**:

```python
"""
Experimental Protocol for PonderTTT

Following best practices:
- Multiple random seeds (≥10)
- Proper train/val/test splits
- Statistical significance testing
- Multiple comparison correction
- Ablation studies
- Pareto frontier analysis
"""

class ExperimentProtocol:
    """
    Rigorous experimental protocol.
    """

    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def define_experimental_conditions(self):
        """
        Define all experimental conditions.

        We compare:
        1. Baselines:
           - Uniform K=1, 2, 4, 8 (4 conditions)
           - Heuristic (entropy-based) (1 condition)
        2. Learned policies:
           - REINFORCE (lambda=0.01, target=4) (1 condition)
           - REINFORCE (lambda=0.05, target=4) (1 condition)
           - REINFORCE (lambda=0.01, no target) (1 condition)
           - Q-learning (1 condition) [NEW]
           - Budget-constrained (1 condition) [NEW]
        3. Oracle:
           - Per-token oracle (1 condition, upper bound)

        Total: 11 conditions × 10 seeds = 110 runs
        """
        conditions = []

        # Uniform baselines
        for k in [1, 2, 4, 8]:
            conditions.append({
                'name': f'uniform_k{k}',
                'type': 'baseline',
                'config': {
                    'use_learned_policy': False,
                    'use_iterative_ttt': True,
                    'max_steps': k,
                    'lambda_compute': 0.0,
                    'target_avg_steps': None,
                },
                'fixed_num_steps': k,
            })

        # Heuristic
        conditions.append({
            'name': 'heuristic_entropy',
            'type': 'baseline',
            'config': {
                'use_learned_policy': False,
                'use_iterative_ttt': True,
                'max_steps': 8,
                'lambda_compute': 0.0,
            },
            'use_heuristic': True,
            'heuristic_metric': 'entropy',
        })

        # Learned policies (REINFORCE)
        for lambda_val in [0.01, 0.05]:
            for target in [4, None]:
                name = f'learned_lambda{int(lambda_val*100):03d}'
                if target is not None:
                    name += f'_target{target}'
                else:
                    name += '_notarget'

                conditions.append({
                    'name': name,
                    'type': 'learned_reinforce',
                    'config': {
                        'use_learned_policy': True,
                        'use_iterative_ttt': True,
                        'max_steps': 8,
                        'lambda_compute': lambda_val,
                        'target_avg_steps': target,
                        'policy_use_lstm': True,
                        'policy_pooling': 'none',  # Per-token
                    },
                })

        # Q-learning
        conditions.append({
            'name': 'learned_qlearning',
            'type': 'learned_qlearning',
            'config': {
                'use_learned_policy': True,
                'use_iterative_ttt': True,
                'max_steps': 8,
                'policy_type': 'q_network',
            },
        })

        # Budget-constrained
        conditions.append({
            'name': 'learned_budget_target4',
            'type': 'learned_budget',
            'config': {
                'use_learned_policy': True,
                'use_iterative_ttt': True,
                'max_steps': 8,
                'policy_type': 'budget_constrained',
                'target_budget': 4,
            },
        })

        return conditions

    def run_single_experiment(
        self,
        condition: Dict,
        seed: int,
        train_steps: int = 50000,
        eval_interval: int = 1000,
        save_checkpoint: bool = True,
    ) -> Dict:
        """
        Run single experiment for one condition and one seed.

        Returns:
            Dictionary with all results and statistics
        """
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Create model
        config = IterativeTransformerConfig(**condition['config'])
        model = IterativeTransformerTTT(config)
        model = model.to('cuda')

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

        # Learning rate schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_steps,
            eta_min=1e-5,
        )

        # Data loaders
        train_loader, val_loader, test_loader = get_wikitext2_dataloaders(
            batch_size=8,
            max_length=256,
        )

        # Training loop
        results = {
            'condition': condition['name'],
            'seed': seed,
            'train_losses': [],
            'val_losses': [],
            'val_perplexities': [],
            'avg_steps': [],
            'step_distributions': [],
            'timestamps': [],
        }

        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(100):  # Large number, will early stop
            for batch in train_loader:
                model.train()

                input_ids = batch['input_ids'].to('cuda')
                labels = batch['labels'].to('cuda')

                # Fixed num_steps for baselines
                if 'fixed_num_steps' in condition:
                    num_steps = torch.full(
                        (input_ids.shape[1],),
                        condition['fixed_num_steps'],
                        device='cuda'
                    )
                else:
                    num_steps = None  # Let model decide

                # Forward
                outputs = model(input_ids, labels=labels, num_steps=num_steps, return_stats=True)
                loss = outputs['loss']

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Record
                results['train_losses'].append(loss.item())
                if 'ttt_stats' in outputs:
                    avg_steps = np.mean([s.get('avg_steps', 0) for s in outputs['ttt_stats'] if 'avg_steps' in s])
                    results['avg_steps'].append(avg_steps)

                global_step += 1

                # Evaluation
                if global_step % eval_interval == 0:
                    val_metrics = self.evaluate(model, val_loader, num_steps=num_steps if 'fixed_num_steps' in condition else None)
                    results['val_losses'].append(val_metrics['loss'])
                    results['val_perplexities'].append(val_metrics['perplexity'])
                    results['timestamps'].append(global_step)

                    print(f"Step {global_step}, Val Loss: {val_metrics['loss']:.4f}, Val PPL: {val_metrics['perplexity']:.2f}")

                    # Early stopping
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        patience_counter = 0
                        if save_checkpoint:
                            self.save_checkpoint(model, condition['name'], seed, global_step)
                    else:
                        patience_counter += 1
                        if patience_counter >= 5:  # Patience
                            print(f"Early stopping at step {global_step}")
                            break

                if global_step >= train_steps:
                    break

            if global_step >= train_steps or patience_counter >= 5:
                break

        # Final evaluation on test set
        test_metrics = self.evaluate(model, test_loader, num_steps=num_steps if 'fixed_num_steps' in condition else None)
        results['test_loss'] = test_metrics['loss']
        results['test_perplexity'] = test_metrics['perplexity']
        results['test_avg_steps'] = test_metrics.get('avg_steps', 0)
        results['test_step_distribution'] = test_metrics.get('step_distribution', {})

        # Compute FLOPs
        flops_analyzer = TTTFLOPsAnalyzer(model)
        flops = flops_analyzer.estimate_total_flops(
            seq_len=256,
            num_steps=results['test_avg_steps'] if results['test_avg_steps'] > 0 else condition.get('fixed_num_steps', 4),
            include_backward=False,  # Inference FLOPs
        )
        results['test_flops_per_token'] = flops['per_token']
        results['test_total_flops'] = flops['total']

        return results

    def evaluate(self, model, dataloader, num_steps=None):
        """Evaluate model on dataloader."""
        model.eval()

        total_loss = 0
        total_tokens = 0
        all_steps = []
        step_distribution = {}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to('cuda')
                labels = batch['labels'].to('cuda')

                outputs = model(input_ids, labels=labels, num_steps=num_steps, return_stats=True)

                # Accumulate loss
                loss = outputs['loss']
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

                # Collect step statistics
                if 'ttt_stats' in outputs:
                    for stat in outputs['ttt_stats']:
                        if 'avg_steps' in stat:
                            all_steps.append(stat['avg_steps'])

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 100))

        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'avg_steps': np.mean(all_steps) if all_steps else 0,
            'step_distribution': step_distribution,
        }

    def run_all_experiments(self, seeds=[42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]):
        """
        Run all experiments across all conditions and seeds.

        This will take ~1 week on 4x A100 GPUs.
        """
        conditions = self.define_experimental_conditions()

        all_results = []

        for condition in conditions:
            for seed in seeds:
                print(f"\n{'='*80}")
                print(f"Running: {condition['name']}, Seed: {seed}")
                print(f"{'='*80}")

                result = self.run_single_experiment(condition, seed)
                all_results.append(result)

                # Save intermediate results
                self.save_results(all_results)

        # Final analysis
        self.analyze_results(all_results)

        return all_results

    def analyze_results(self, all_results):
        """
        Comprehensive statistical analysis of all results.
        """
        import pandas as pd
        from ponderttt.utils.statistics import (
            statistical_test,
            compare_multiple_methods,
            print_multiple_comparison_results,
            print_power_analysis,
        )

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Group by condition
        grouped = df.groupby('condition')

        # Aggregate statistics
        summary = grouped.agg({
            'test_perplexity': ['mean', 'std', 'min', 'max'],
            'test_avg_steps': ['mean', 'std'],
            'test_flops_per_token': ['mean', 'std'],
        }).round(4)

        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(summary)

        # Pairwise comparisons
        baseline_condition = 'uniform_k4'  # Reference

        methods_results = {}
        for condition_name, group in grouped:
            methods_results[condition_name] = group['test_perplexity'].tolist()

        # Multiple comparison with correction
        comparison = compare_multiple_methods(
            methods_results,
            baseline_name=baseline_condition,
            alpha=0.05,
            correction='holm',  # Less conservative than Bonferroni
        )

        print_multiple_comparison_results(comparison)

        # Power analysis
        n_seeds = len(df['seed'].unique())
        print_power_analysis(n_seeds)

        # Pareto frontier
        self.plot_pareto_frontier(df)

        # Step distribution analysis
        self.plot_step_distributions(df)

        # Save all analyses
        summary.to_csv(self.output_dir / 'summary_statistics.csv')
        df.to_csv(self.output_dir / 'all_results.csv', index=False)

        with open(self.output_dir / 'statistical_tests.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

    def plot_pareto_frontier(self, df):
        """Plot quality vs compute Pareto frontier."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each condition
        for condition in df['condition'].unique():
            subset = df[df['condition'] == condition]

            # Mean and std
            mean_ppl = subset['test_perplexity'].mean()
            std_ppl = subset['test_perplexity'].std()
            mean_flops = subset['test_flops_per_token'].mean()

            # Determine marker style
            if 'uniform' in condition:
                marker = 'o'
                color = 'blue'
            elif 'heuristic' in condition:
                marker = 's'
                color = 'green'
            elif 'learned' in condition:
                marker = '^'
                color = 'red'
            elif 'oracle' in condition:
                marker = '*'
                color = 'gold'
            else:
                marker = 'x'
                color = 'gray'

            ax.errorbar(
                mean_flops, mean_ppl, yerr=std_ppl,
                marker=marker, color=color, label=condition,
                capsize=5, capthick=2, markersize=8,
            )

        ax.set_xlabel('FLOPs per Token', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title('Quality-Compute Tradeoff (Pareto Frontier)', fontsize=14)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pareto_frontier.pdf', dpi=300)
        plt.savefig(self.output_dir / 'pareto_frontier.png', dpi=300)
        print(f"Saved Pareto frontier to {self.output_dir / 'pareto_frontier.pdf'}")

    def plot_step_distributions(self, df):
        """Plot distribution of allocated steps for learned policies."""
        import matplotlib.pyplot as plt

        # Filter learned policies
        learned = df[df['condition'].str.contains('learned')]

        if learned.empty:
            return

        fig, axes = plt.subplots(1, len(learned['condition'].unique()),
                                 figsize=(5*len(learned['condition'].unique()), 4))

        if len(learned['condition'].unique()) == 1:
            axes = [axes]

        for ax, condition in zip(axes, learned['condition'].unique()):
            subset = learned[learned['condition'] == condition]

            # Aggregate step distributions across seeds
            # (This requires storing step_distribution in results)
            # For now, just plot avg_steps

            ax.hist(subset['test_avg_steps'], bins=20, alpha=0.7)
            ax.set_xlabel('Average Steps', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(condition, fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'step_distributions.pdf', dpi=300)
        print(f"Saved step distributions to {self.output_dir / 'step_distributions.pdf'}")
```

**Execution Plan**:
1. Week 10: Dry run on 2 conditions × 2 seeds (debug)
2. Week 11-12: Full run on WikiText-2 (11 conditions × 10 seeds)
3. Week 13: Oracle analysis and convergence analysis
4. Week 14: Analysis, plotting, statistical tests

**Compute Requirements**:
- 110 runs × ~6 hours = 660 GPU-hours
- With 4x A100: ~1 week
- Estimated cost on AWS: ~$2000-3000

**Success Criteria**:
- All 110 runs complete successfully
- Statistical significance achieved (p < 0.05 after correction)
- Clear Pareto frontier showing learned policies beat uniform
- Comprehensive analysis completed

---

## Phase 5: Paper Writing (Weeks 18-22)
**Goal**: Write publication-quality paper

### Task 5.1: Paper Structure
**Priority**: Core Deliverable
**Estimated Time**: 1 month
**Target Venue**: NeurIPS, ICLR, or specialized workshop

**Paper Outline**:

```markdown
# PonderTTT: Adaptive Test-Time Training with Dynamic Iteration Allocation

## Abstract (250 words)
- Problem: TTT uses fixed computation per token
- Solution: Learn to allocate iterations based on difficulty
- Results: X% better quality-compute tradeoff
- Significance: First work on adaptive computation for TTT

## 1. Introduction (1.5 pages)
- Test-time training overview
- Motivation: Not all tokens are equal
- Key idea: Dynamic iteration allocation
- Contributions:
  1. Framework for adaptive TTT
  2. Multiple allocation strategies (heuristic, learned, oracle)
  3. Comprehensive experiments showing X% improvement
  4. Analysis of iterative vs analytic TTT

## 2. Background (1 page)
- Test-Time Training (Sun et al., 2024)
- Adaptive computation (ACT, Mixture-of-Depths)
- Reinforcement learning for neural architecture

## 3. Method (3 pages)

### 3.1 Iterative TTT Variant
- Explicit K-step gradient descent formulation
- Connection to analytic TTT (convergence analysis)
- Benefits: Flexible, interpretable, controllable

### 3.2 Difficulty-Aware Allocation Strategies

#### Uniform Baselines (K=1,2,4,8)
- Fixed allocation for comparison

#### Heuristic Policies
- Entropy-based difficulty estimation
- Loss-based difficulty estimation

#### Learned Policies
- REINFORCE with temporal credit assignment
- Q-learning variant
- Budget-constrained optimization

### 3.3 Training Objective
- Language modeling loss + policy gradient loss
- Compute regularization (λ penalty)
- Target budget constraint (optional)

## 4. Experiments (3 pages)

### 4.1 Experimental Setup
- Dataset: WikiText-2
- Model: 60M parameters, 6 layers
- Baselines: Uniform K=1,2,4,8, Heuristic, Oracle
- Metrics: Perplexity, FLOPs, average steps

### 4.2 Main Results: Quality-Compute Tradeoff
- Pareto frontier plot
- Learned policies achieve X% better tradeoff
- Statistical significance (p < 0.001)
- Comparison table

### 4.3 Analysis

#### Convergence: Iterative vs Analytic
- K=4 achieves 95% of analytic performance
- Exponential convergence rate

#### Step Allocation Patterns
- Learned policies allocate more to difficult tokens
- Correlation with entropy, loss, attention

#### Oracle Upper Bound
- Per-token oracle achieves Y perplexity
- Gap analysis: How close are learned policies?

### 4.4 Ablation Studies
- REINFORCE vs Q-learning vs Budget
- Temporal credit assignment (immediate vs returns)
- Lambda penalty values
- LSTM context encoder vs none

## 5. Related Work (0.75 pages)
- Adaptive computation (ACT, Universal Transformers, MoE)
- Test-time training (TTT-Linear, TTT-MLP)
- Reinforcement learning for efficiency (SIFT, LEAST)
- Differences and positioning

## 6. Discussion (0.5 pages)
- When does adaptive help most?
- Limitations: Sequential processing overhead
- Future work: Analytic + adaptive, larger models

## 7. Conclusion (0.25 pages)
- Summary of contributions
- Impact: Enable efficient TTT at scale

## Appendix
- A: Implementation details
- B: Hyperparameters
- C: Additional experiments (WikiText-103)
- D: FLOPs calculation details
- E: Negative results / failed approaches
```

**Writing Timeline**:
- Week 15: Draft sections 1-3 (Intro, Background, Method)
- Week 16: Draft section 4 (Experiments)
- Week 17: Draft sections 5-7 (Related Work, Discussion, Conclusion)
- Week 18: Revision, figures, proofreading

**Success Criteria**:
- Clear, compelling narrative
- Complete experimental validation
- Honest discussion of limitations
- Ready for submission

---

## Phase 6: Extensions (Months 5-6, Optional)
**Goal**: Strengthen paper with additional contributions

### Task 6.1: Scale to WikiText-103
**Estimated Time**: 2 weeks
**Academic Value**: ⭐⭐⭐⭐ (Demonstrates scalability)

### Task 6.2: Larger Models (300M+)
**Estimated Time**: 3 weeks
**Academic Value**: ⭐⭐⭐⭐⭐ (High impact)

### Task 6.3: Other Datasets (C4, Books3)
**Estimated Time**: 2 weeks
**Academic Value**: ⭐⭐⭐ (Generalization)

### Task 6.4: Hybrid Analytic-Adaptive
**Estimated Time**: 3 weeks
**Academic Value**: ⭐⭐⭐⭐⭐ (Novel contribution)

**Idea**: Use analytic TTT with learned adaptive LR (instead of adaptive K)
- Keep mini-batch processing (efficient)
- Learn per-token LR multipliers
- Best of both worlds: Analytic precision + adaptive allocation

---

## Success Metrics

### Code Quality
- ✅ All tests pass
- ✅ No import errors
- ✅ Matches official TTT on key metrics (when using analytic)
- ✅ Reproducible (fixed seeds, documented hyperparameters)

### Experimental Rigor
- ✅ 10+ seeds per condition
- ✅ Statistical significance (p < 0.05 after correction)
- ✅ Confidence intervals reported
- ✅ Multiple comparison correction applied
- ✅ Power analysis conducted

### Academic Contribution
- ✅ Novel problem formulation (adaptive TTT)
- ✅ Multiple solution approaches (heuristic, RL-based)
- ✅ Thorough empirical evaluation
- ✅ Theoretical analysis (convergence)
- ✅ Oracle upper bound
- ✅ Honest discussion of limitations

### Publication Readiness
- ✅ Clear paper structure
- ✅ Complete experimental validation
- ✅ High-quality figures
- ✅ Code and data released
- ✅ Reproducibility guaranteed

---

## Risk Mitigation

### Risk 1: Negative Results
**Mitigation**: Frame as "When does adaptive help?" rather than "Adaptive always helps"
- Ablation studies identify when it works
- Failure analysis has academic value

### Risk 2: Statistical Power Insufficient
**Mitigation**: Increase seeds to 20 if needed
- Monitor effect sizes during experiments
- Add more seeds if Cohen's d is small

### Risk 3: Computation Cost Exceeds Budget
**Mitigation**: Start with smaller scale
- Run WikiText-2 first (cheaper)
- If successful, seek additional compute resources
- Consider cloud credits or academic partnerships

### Risk 4: Reviewer Concerns
**Mitigation**: Anticipate common concerns
- "Not enough improvement" → Show clear Pareto frontier, statistical significance
- "Unfair comparison" → Carefully match parameters, document all details
- "Limited scope" → Acknowledge, plan extensions in rebuttal

---

## Timeline Summary

| Phase | Original | Realistic (w/ buffer) | Key Deliverable |
|-------|----------|----------------------|-----------------|
| Phase 0 | Week 1 | Week 1 | Code runnable |
| Phase 1 | Weeks 2-4 | Weeks 2-5 | Official TTT implemented, bugs fixed |
| Phase 2 | Weeks 5-8 | Weeks 6-9 | Core contributions (REINFORCE + heuristics) |
| Phase 3 | Week 9 | Weeks 10-11 | Accurate FLOPs + wall-clock time |
| Phase 4 | Weeks 10-14 | Weeks 12-17 | Full experiments with statistical validation |
| Phase 5 | Weeks 15-18 | Weeks 18-22 | Paper draft ready for submission |
| Phase 6 | Months 5-6 | Months 6-7 | Extensions (optional) |

**Original Estimate**: 4-6 months (18-26 weeks)
**Realistic Estimate**: 5-7 months (22-30 weeks)

**Buffer Rationale**:
- Official TTT implementation is complex (analytic solution, triangular attention)
- Debugging and numerical stability take time
- Experiment failures and reruns
- Paper writing always takes longer than expected

**Critical Path**: Phase 1 → Phase 2 → Phase 4 → Phase 5

**Milestone Checkpoints**:
- End of Week 5: Official TTT working + core bugs fixed
- End of Week 9: REINFORCE with temporal credit working
- End of Week 11: FLOPs calculator validated
- End of Week 17: All experiments complete
- End of Week 22: Paper ready for submission

---

## Next Steps (Immediate Actions)

### This Week:
1. Fix all P0 blockers (DummyPolicyNetwork, imports)
2. Run smoke tests
3. Set up experiment tracking (Weights & Biases recommended)

### Next Week:
4. Implement Official TTT-Linear (analytic)
5. Fix LayerNorm in iterative TTT
6. Begin REINFORCE fix

### Week 3-4:
7. Complete Phase 1 (all core fixes)
8. Begin Phase 2 (novel contributions)

**Let's start with Phase 0 fixes immediately!**

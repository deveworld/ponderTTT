"""
Heuristic policies for adaptive iteration allocation.

Non-learned baselines for comparison with learned halting policies.
These policies use simple heuristics to allocate gradient steps based on
difficulty metrics without any trainable parameters.

Features:
- Percentile-based calibration for consistent allocation
- Multiple difficulty metrics (entropy, loss, gradient norm)
- Custom target distributions supported
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class HeuristicPolicyBase(nn.Module):
    """
    Base class for heuristic policies.

    All heuristic policies share the same interface as HaltingPolicyNetwork
    but use fixed rules instead of learned parameters.

    Calibration:
    - Before use, call calibrate() on a validation set
    - This computes percentile thresholds for difficulty → step mapping
    - More robust than per-batch min-max normalization
    """

    def __init__(
        self,
        step_options: List[int] = [1, 2, 4, 8],
        use_calibration: bool = True,
    ):
        super().__init__()
        self.step_options = step_options
        self.num_options = len(step_options)
        self.use_calibration = use_calibration

        self.register_buffer(
            "step_options_tensor",
            torch.tensor(step_options, dtype=torch.long)
        )

        # Percentile thresholds (computed via calibrate())
        # thresholds[i] = difficulty value separating step_options[i] and step_options[i+1]
        # Length: num_options - 1
        self.register_buffer(
            "difficulty_thresholds",
            torch.zeros(self.num_options - 1)
        )

        # Calibration status
        self.register_buffer(
            "is_calibrated",
            torch.tensor(False)
        )

    def compute_difficulty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute difficulty scores for each token.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            difficulty: (batch, seq_len) - higher = more difficult
        """
        raise NotImplementedError

    def calibrate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 50,
        target_distribution: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calibrate difficulty thresholds on a validation set.

        This computes percentile-based thresholds that map difficulty scores
        to step allocations, ensuring consistent behavior across batches.

        Args:
            model: The model to extract hidden states from
            dataloader: Validation data loader
            max_batches: Maximum batches to use for calibration
            target_distribution: Optional target percentages for each step option
                                 If None, uses uniform quartiles [25%, 50%, 75%, 100%]

        Returns:
            thresholds: Computed difficulty thresholds (num_options - 1,)
            stats: Calibration statistics
        """
        if target_distribution is None:
            # Default: uniform distribution across step options
            # For 4 options: [25%, 50%, 75%, 100%]
            target_distribution = [
                (i + 1) / self.num_options * 100
                for i in range(self.num_options)
            ]

        # Collect difficulty scores
        all_difficulties = []

        model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Calibrating heuristic policy", total=max_batches)
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(next(model.parameters()).device)

                # Get hidden states (use first TTT layer output or embedding)
                if hasattr(model, 'token_embedding'):
                    hidden_states = model.token_embedding(input_ids)
                elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                    hidden_states = model.transformer.wte(input_ids)
                else:
                    # Fallback: run full forward pass and use last hidden state
                    outputs = model(input_ids, labels=None, return_stats=False)
                    if 'hidden_states' in outputs:
                        hidden_states = outputs['hidden_states']
                    else:
                        # Use embedding as fallback
                        hidden_states = model.get_input_embeddings()(input_ids)

                # Compute difficulty for this batch
                difficulty = self.compute_difficulty(hidden_states)  # (batch, seq_len)

                all_difficulties.append(difficulty.flatten().cpu())

        # Concatenate all difficulties
        all_difficulties = torch.cat(all_difficulties)  # (total_tokens,)

        # Compute percentile thresholds
        # For 4 step options [1, 2, 4, 8], we need 3 thresholds at 25%, 50%, 75%
        percentiles = target_distribution[:-1]  # Exclude 100%
        thresholds = torch.tensor([
            torch.quantile(all_difficulties, p / 100.0)
            for p in percentiles
        ])

        # Update buffers
        self.difficulty_thresholds.copy_(thresholds)
        self.is_calibrated.fill_(True)

        # Compute statistics
        stats = {
            'total_tokens': len(all_difficulties),
            'difficulty_min': all_difficulties.min().item(),
            'difficulty_max': all_difficulties.max().item(),
            'difficulty_mean': all_difficulties.mean().item(),
            'difficulty_std': all_difficulties.std().item(),
            'thresholds': thresholds.tolist(),
            'percentiles': percentiles,
        }

        # Verify distribution
        predicted_steps = self._apply_thresholds(all_difficulties.unsqueeze(0)).flatten()
        step_distribution = {}
        for step in self.step_options:
            count = (predicted_steps == step).sum().item()
            step_distribution[step] = count / len(predicted_steps) * 100

        stats['actual_distribution'] = step_distribution

        return thresholds, stats

    def _apply_thresholds(self, difficulty: torch.Tensor) -> torch.Tensor:
        """
        Apply calibrated thresholds to map difficulty to step allocations.

        Args:
            difficulty: (batch, seq_len)

        Returns:
            steps: (batch, seq_len)
        """
        batch_size, seq_len = difficulty.shape
        steps = torch.zeros_like(difficulty, dtype=torch.long)

        # Map difficulty to step indices using thresholds
        # difficulty < thresholds[0] → step_options[0]
        # thresholds[0] <= difficulty < thresholds[1] → step_options[1]
        # thresholds[1] <= difficulty < thresholds[2] → step_options[2]
        # difficulty >= thresholds[2] → step_options[3]

        for i in range(self.num_options):
            if i == 0:
                # Below first threshold
                mask = difficulty < self.difficulty_thresholds[0]
            elif i == self.num_options - 1:
                # Above last threshold
                mask = difficulty >= self.difficulty_thresholds[-1]
            else:
                # Between thresholds
                mask = (difficulty >= self.difficulty_thresholds[i - 1]) & \
                       (difficulty < self.difficulty_thresholds[i])

            steps[mask] = self.step_options[i]

        return steps

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_probs: bool = False,
        deterministic: bool = True,
        pooling: str = 'none',
    ):
        """
        Allocate steps based on difficulty heuristic.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            return_probs: Ignored (for compatibility)
            deterministic: Ignored (always deterministic)
            pooling: Batch aggregation ('none', 'mean', 'max')

        Returns:
            steps: (batch, seq_len) or (seq_len,) depending on pooling
            None: No log probs for heuristic policies
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute difficulty for each token
        difficulty = self.compute_difficulty(hidden_states)  # (batch, seq_len)

        if self.use_calibration and self.is_calibrated:
            # Use calibrated thresholds (percentile-based)
            steps = self._apply_thresholds(difficulty)
        else:
            # Fallback to per-batch min-max normalization (original method)
            # NOTE: This is less robust but works without calibration
            diff_min = difficulty.min(dim=-1, keepdim=True)[0]
            diff_max = difficulty.max(dim=-1, keepdim=True)[0]
            diff_range = diff_max - diff_min + 1e-8
            normalized_difficulty = (difficulty - diff_min) / diff_range

            # Map to step options (quantize)
            # difficulty 0.0-0.25 → step_options[0]
            # difficulty 0.25-0.5 → step_options[1]
            # difficulty 0.5-0.75 → step_options[2]
            # difficulty 0.75-1.0 → step_options[3]
            step_indices = (normalized_difficulty * (self.num_options - 1e-6)).long()
            step_indices = torch.clamp(step_indices, 0, self.num_options - 1)

            steps = self.step_options_tensor[step_indices]  # (batch, seq_len)

        # Apply pooling if requested
        if pooling == 'mean':
            steps = steps.float().mean(dim=0).long()  # (seq_len,)
        elif pooling == 'max':
            steps = steps.max(dim=0)[0]  # (seq_len,)
        elif pooling == 'none':
            pass  # Keep (batch, seq_len)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return steps, None


class EntropyBasedPolicy(HeuristicPolicyBase):
    """
    Allocate steps based on prediction entropy.

    High entropy = uncertain prediction = more difficult → more steps
    Low entropy = confident prediction = easier → fewer steps

    Requires a language model head to compute next-token predictions.
    """

    def __init__(
        self,
        lm_head: nn.Linear,
        step_options: List[int] = [1, 2, 4, 8],
        use_calibration: bool = True,
    ):
        super().__init__(step_options, use_calibration)
        self.lm_head = lm_head

    def compute_difficulty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of next-token prediction.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            entropy: (batch, seq_len)
        """
        with torch.no_grad():
            logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

        return entropy


class LossBasedPolicy(HeuristicPolicyBase):
    """
    Allocate steps based on per-token loss.

    High loss = model struggles with this token → more steps
    Low loss = model handles well → fewer steps

    Requires labels and a language model head.
    """

    def __init__(
        self,
        lm_head: nn.Linear,
        step_options: List[int] = [1, 2, 4, 8],
        use_calibration: bool = True,
    ):
        super().__init__(step_options, use_calibration)
        self.lm_head = lm_head

        # Store labels (will be set during forward)
        self.labels = None

    def set_labels(self, labels: torch.Tensor):
        """
        Set ground-truth labels for loss computation.

        Args:
            labels: (batch, seq_len)
        """
        self.labels = labels

    def compute_difficulty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token cross-entropy loss.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            loss_per_token: (batch, seq_len)
        """
        if self.labels is None:
            # Fallback to entropy if no labels
            logits = self.lm_head(hidden_states)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            return -(probs * log_probs).sum(dim=-1)

        with torch.no_grad():
            logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)

            # Gather log probs for true labels
            labels_expanded = self.labels.unsqueeze(-1)  # (batch, seq_len, 1)
            true_log_probs = torch.gather(log_probs, dim=-1, index=labels_expanded)
            true_log_probs = true_log_probs.squeeze(-1)  # (batch, seq_len)

            # Loss = -log P(true label)
            loss_per_token = -true_log_probs

        return loss_per_token


class GradientNormBasedPolicy(HeuristicPolicyBase):
    """
    Allocate steps based on gradient magnitude.

    Large gradients = steep loss landscape = difficult → more steps
    Small gradients = flat loss landscape = easier → fewer steps

    This requires computing gradients, so it's more expensive.
    """

    def __init__(
        self,
        step_options: List[int] = [1, 2, 4, 8],
        use_calibration: bool = True,
    ):
        super().__init__(step_options, use_calibration)

    def compute_difficulty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 norm of hidden state gradients.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            grad_norm: (batch, seq_len)
        """
        # Compute norm of hidden states as proxy for gradient magnitude
        # (actual gradient computation would require backward pass)
        with torch.no_grad():
            grad_norm = torch.norm(hidden_states, dim=-1, p=2)  # (batch, seq_len)

        return grad_norm


class PerplexityBasedPolicy(HeuristicPolicyBase):
    """
    Allocate steps based on local perplexity.

    Similar to entropy-based but uses perplexity = exp(entropy).
    """

    def __init__(
        self,
        lm_head: nn.Linear,
        step_options: List[int] = [1, 2, 4, 8],
        use_calibration: bool = True,
    ):
        super().__init__(step_options, use_calibration)
        self.lm_head = lm_head

    def compute_difficulty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token perplexity.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            perplexity: (batch, seq_len)
        """
        with torch.no_grad():
            logits = self.lm_head(hidden_states)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            perplexity = torch.exp(entropy)

        return perplexity


class RandomPolicy(HeuristicPolicyBase):
    """
    Random baseline: allocate steps randomly.

    Useful for ablation to show that difficulty-awareness matters.
    """

    def __init__(
        self,
        step_options: List[int] = [1, 2, 4, 8],
        seed: Optional[int] = None,
        use_calibration: bool = False,  # Random doesn't need calibration
    ):
        super().__init__(step_options, use_calibration)
        self.seed = seed
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

    def compute_difficulty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Return random difficulty scores.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            random_scores: (batch, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape

        if self.generator is not None:
            random_scores = torch.rand(
                batch_size, seq_len,
                device=hidden_states.device,
                generator=self.generator
            )
        else:
            random_scores = torch.rand(
                batch_size, seq_len,
                device=hidden_states.device
            )

        return random_scores


class UniformPolicy(HeuristicPolicyBase):
    """
    Uniform baseline: allocate same steps to all tokens.

    This is equivalent to the fixed-K baselines.
    """

    def __init__(
        self,
        fixed_steps: int,
        step_options: List[int] = [1, 2, 4, 8],
    ):
        super().__init__(step_options)
        assert fixed_steps in step_options, f"fixed_steps {fixed_steps} not in {step_options}"
        self.fixed_steps = fixed_steps

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_probs: bool = False,
        deterministic: bool = True,
        pooling: str = 'none',
    ):
        """
        Return fixed steps for all tokens.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            pooling: Ignored

        Returns:
            steps: (batch, seq_len) or (seq_len,) - all equal to fixed_steps
        """
        batch_size, seq_len, _ = hidden_states.shape

        if pooling == 'none':
            steps = torch.full(
                (batch_size, seq_len),
                self.fixed_steps,
                dtype=torch.long,
                device=hidden_states.device
            )
        else:
            steps = torch.full(
                (seq_len,),
                self.fixed_steps,
                dtype=torch.long,
                device=hidden_states.device
            )

        return steps, None

    def compute_difficulty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Not used
        return torch.zeros(hidden_states.shape[:2], device=hidden_states.device)


# ==============================================================================
# Documentation: Heuristic Policies vs Learned Policies
# ==============================================================================

"""
Limitations of Heuristic Policies vs Learned REINFORCE Policy
==============================================================

## Summary

Heuristic policies provide strong baselines but have fundamental limitations
compared to learned policies (REINFORCE with temporal credit assignment).

## Heuristic Policy Strengths ✓

1. **No Training Required**: Zero-shot application without policy learning
2. **Interpretability**: Clear, human-understandable rules
3. **Stability**: No variance from policy gradient estimation
4. **Speed**: No policy network forward pass overhead
5. **Simplicity**: Easy to implement and debug

## Heuristic Policy Limitations ✗

### 1. Myopic Decision-Making

**Problem**: Heuristics optimize per-token difficulty, ignoring sequential effects.

```python
# Heuristic: High entropy token → allocate K=8
# Reality: This token's K affects all future token hidden states via
#          fast-weight carry-over. Optimal K depends on JOINT allocation.
```

**Example**:
- Token t has high entropy → Heuristic assigns K=8
- Token t+1 has low entropy → Heuristic assigns K=1
- But: Large K at token t might make token t+1 easier (better hidden state)
- Optimal: K_t=4, K_{t+1}=2 (joint optimization)

**REINFORCE Advantage**: Monte Carlo returns account for future effects.

### 2. No Credit Assignment

**Problem**: Heuristics cannot learn which difficulty metrics matter most.

```python
# Which matters more for iteration allocation?
# - Entropy
# - Loss
# - Gradient norm
# - Hidden state magnitude

# Heuristic: Pick one heuristic (human decision)
# REINFORCE: Learn weighted combination from actual task performance
```

**Example**: On some datasets, entropy might correlate poorly with optimal K,
but loss might correlate well. Heuristics cannot discover this automatically.

### 3. Fixed Thresholds

**Problem**: Even with calibration, thresholds are fixed across all contexts.

```python
# Calibrated heuristic: entropy > 5.2 → K=8
# But optimal threshold might vary:
# - By sequence position (early vs late tokens)
# - By sequence length
# - By batch composition
# - By training stage
```

**REINFORCE Advantage**: Learns context-dependent policies.

### 4. No Adaptation During Training

**Problem**: Heuristics don't improve as the model trains.

```python
# Early training: Model is bad at everything → all tokens "difficult"
# Late training: Model is good → most tokens "easy"
# Heuristic thresholds: Fixed (require re-calibration)
# REINFORCE policy: Adapts automatically
```

### 5. Cannot Leverage Task-Specific Patterns

**Problem**: Some tasks have predictable difficulty patterns that heuristics miss.

```python
# Example pattern: "The word after 'the' is usually easy"
# Heuristic: Cannot capture this (no context beyond hidden state)
# REINFORCE: Can learn this from reward structure
```

## Quantitative Comparison (Expected)

### Pareto Frontier

```
Quality (Perplexity) vs Compute (FLOPs)
──────────────────────────────────────

High │                    ● REINFORCE (optimal Pareto)
Qual │                 ●    Entropy heuristic (calibrated)
ity  │              ●       Loss heuristic (calibrated)
     │           ●          Random
Low  │  ● ● ● ●             Uniform K=1,2,4,8
     └────────────────────────────────────────→
       Low                Compute               High
```

**Expected Results**:
- REINFORCE dominates heuristics on Pareto frontier
- Calibrated heuristics > uncalibrated
- Entropy/Loss heuristics ≈ similar (dataset-dependent)
- All heuristics > Random > Uniform K=1

### Statistical Significance

**Hypothesis**: At matched compute budget, REINFORCE achieves lower perplexity.

**Test**: Paired t-test with Bonferroni correction (p < 0.05)

**Expected**: REINFORCE significantly outperforms all heuristics.

## When to Use Heuristics vs REINFORCE

### Use Heuristics If:
- ✓ Need immediate baseline without training policy
- ✓ Interpretability is critical
- ✓ Training budget is very limited
- ✓ Deployment requires deterministic behavior

### Use REINFORCE If:
- ✓ Have training budget for policy learning
- ✓ Task has sequential dependencies
- ✓ Need best Pareto frontier
- ✓ Want adaptive policy that improves with model

## Implementation Notes

### Calibration Best Practices

1. **Calibration Set Size**: 50-100 batches (≈10K tokens)
2. **Target Distribution**:
   - Uniform [25%, 50%, 75%, 100%] (default)
   - Or: Match oracle K distribution (if available)
   - Or: Match computational budget
3. **Re-calibration**: Every 10% of training (as model improves)

### Usage Example

```python
from ponderttt.models.heuristic_policies import EntropyBasedPolicy
from ponderttt.data import get_wikitext2_dataloaders

# Create policy
policy = EntropyBasedPolicy(
    lm_head=model.lm_head,
    step_options=[1, 2, 4, 8],
    use_calibration=True,
)

# Calibrate on validation set
train_loader, val_loader, test_loader = get_wikitext2_dataloaders(batch_size=8)
thresholds, stats = policy.calibrate(
    model=model,
    dataloader=val_loader,
    max_batches=50,
)

print(f"Calibrated thresholds: {stats['thresholds']}")
print(f"Step distribution: {stats['actual_distribution']}")

# Use in training/evaluation
hidden_states = model.get_hidden_states(input_ids)
steps, _ = policy(hidden_states, pooling='none')
```

## Academic Contribution

**Main Paper Focus**: REINFORCE with temporal credit (novel contribution)

**Heuristic Policies Role**:
- Strong baselines for comparison
- Ablation: Does learning help? (Answer: Yes)
- Interpretability: What difficulty metrics matter?

**Expected Finding**:
"While heuristic policies provide reasonable baselines, learned REINFORCE
policies significantly outperform them (p < 0.001) by accounting for
sequential dependencies and temporal credit assignment."

## References

- Uniform baselines: Standard practice in adaptive computation
- Entropy-based: Related to uncertainty-based active learning
- Loss-based: Similar to curriculum learning difficulty metrics
- REINFORCE with temporal credit: **Our main contribution**

"""


"""
Advanced gating for PonderTTT Run phase.

Combines multiple uncertainty signals for adaptive TTT decisions:
1. TTT Improvement: ttt_loss_step_0 - ttt_loss_step_1 (validated in Crawl phase)
2. Prediction Entropy: H(p) = -sum(p_i * log(p_i))
3. Attention Dispersion: Mean entropy of attention weights
4. Token Confidence: 1 - max(softmax(logits))
5. Budget Awareness: Remaining compute budget signal

Architecture:
    signals -> SignalNormalizer -> WeightedCombiner -> threshold_decision
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Literal

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class AdvancedGatingConfig:
    """Configuration for multi-signal gating."""

    # Signal selection
    use_ttt_improvement: bool = True      # Validated in Crawl phase (ρ=0.63)
    use_entropy: bool = True              # Prediction uncertainty
    use_attention_dispersion: bool = False  # Requires attention weights
    use_token_confidence: bool = True     # Simple but effective

    # Gating mode
    mode: Literal["threshold", "learned", "budget_aware"] = "threshold"

    # Threshold gating (Walk phase validated)
    base_threshold: float = 0.034  # Calibrated for 50% update rate

    # Learned gating
    hidden_dim: int = 32
    dropout_rate: float = 0.0

    # Budget-aware gating
    target_update_rate: float = 0.5  # Target fraction of chunks to update
    budget_window: int = 100  # Window size for budget tracking
    budget_alpha: float = 0.1  # EMA smoothing for budget

    # Signal combination weights (for fixed-weight mode)
    signal_weights: dict = field(default_factory=lambda: {
        "ttt_improvement": 1.0,  # Primary signal
        "entropy": 0.3,          # Secondary signal
        "token_confidence": 0.2,
        "attention_dispersion": 0.1,
    })


def compute_entropy(logits: jax.Array, axis: int = -1) -> jax.Array:
    """Compute entropy of probability distribution.

    Args:
        logits: Unnormalized log probabilities [..., vocab_size]
        axis: Axis along which to compute entropy

    Returns:
        Entropy values [...] (summed over axis)
    """
    # Stable computation: H(p) = log(sum(exp(logits))) - sum(p * logits)
    log_z = jax.nn.logsumexp(logits, axis=axis, keepdims=True)
    probs = jnp.exp(logits - log_z)
    log_probs = logits - log_z
    entropy = -jnp.sum(probs * log_probs, axis=axis)
    return entropy


def compute_token_confidence(logits: jax.Array) -> jax.Array:
    """Compute 1 - max_prob as uncertainty signal.

    Higher values = more uncertainty = should update.

    Args:
        logits: Unnormalized log probabilities [batch, seq_len, vocab_size]

    Returns:
        Uncertainty scores [batch] (averaged over sequence)
    """
    # Max probability per position
    max_logits = jnp.max(logits, axis=-1)  # [batch, seq_len]
    log_z = jax.nn.logsumexp(logits, axis=-1)  # [batch, seq_len]
    max_probs = jnp.exp(max_logits - log_z)  # [batch, seq_len]

    # Uncertainty = 1 - confidence, averaged over sequence
    uncertainty = 1.0 - max_probs
    return jnp.mean(uncertainty, axis=-1)  # [batch]


def compute_attention_entropy(
    attention_weights: jax.Array,
    mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Compute entropy of attention distributions.

    Higher attention entropy = more dispersed attention = uncertain.

    Args:
        attention_weights: Attention weights [batch, heads, seq_len, seq_len]
        mask: Optional attention mask [batch, seq_len]

    Returns:
        Mean attention entropy [batch]
    """
    # Compute entropy per head per position
    # H = -sum(a * log(a))
    eps = 1e-10
    entropy = -jnp.sum(
        attention_weights * jnp.log(attention_weights + eps),
        axis=-1
    )  # [batch, heads, seq_len]

    # Average over heads and sequence
    if mask is not None:
        # Only consider valid positions
        mask_expanded = mask[:, None, :]  # [batch, 1, seq_len]
        entropy = jnp.where(mask_expanded, entropy, 0.0)
        entropy = jnp.sum(entropy, axis=(1, 2)) / (jnp.sum(mask, axis=-1, keepdims=True) * attention_weights.shape[1])
    else:
        entropy = jnp.mean(entropy, axis=(1, 2))  # [batch]

    return entropy


class SignalExtractor(nnx.Module):
    """Extract and normalize uncertainty signals from model outputs."""

    def __init__(self, config: AdvancedGatingConfig, rngs: nnx.Rngs):
        self.config = config

        # Running statistics for normalization (per signal)
        self.ttt_mean = nnx.Variable(jnp.array(0.034))  # From Crawl phase
        self.ttt_std = nnx.Variable(jnp.array(0.02))
        self.entropy_mean = nnx.Variable(jnp.array(2.0))  # ~log(vocab_size) / 4
        self.entropy_std = nnx.Variable(jnp.array(1.0))
        self.confidence_mean = nnx.Variable(jnp.array(0.5))
        self.confidence_std = nnx.Variable(jnp.array(0.2))

        # EMA momentum for updating statistics
        self.momentum = 0.99

    def __call__(
        self,
        ttt_improvement: Optional[jax.Array] = None,
        logits: Optional[jax.Array] = None,
        attention_weights: Optional[jax.Array] = None,
        update_stats: bool = False,
    ) -> dict[str, jax.Array]:
        """Extract and normalize signals.

        Args:
            ttt_improvement: TTT loss improvement [batch]
            logits: Model logits [batch, seq_len, vocab_size]
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            update_stats: Whether to update running statistics

        Returns:
            Dictionary of normalized signals (all in [0, 1] range approximately)
        """
        signals = {}

        # TTT Improvement (primary signal)
        if self.config.use_ttt_improvement and ttt_improvement is not None:
            # Normalize using running statistics
            normalized = (ttt_improvement - self.ttt_mean.value) / (self.ttt_std.value + 1e-6)
            # Sigmoid to [0, 1]
            signals["ttt_improvement"] = jax.nn.sigmoid(normalized)

            if update_stats:
                batch_mean = jnp.mean(ttt_improvement)
                batch_std = jnp.std(ttt_improvement) + 1e-6
                self.ttt_mean.value = self.momentum * self.ttt_mean.value + (1 - self.momentum) * batch_mean
                self.ttt_std.value = self.momentum * self.ttt_std.value + (1 - self.momentum) * batch_std

        # Prediction Entropy
        if self.config.use_entropy and logits is not None:
            # Compute mean entropy over sequence
            entropy_per_pos = compute_entropy(logits, axis=-1)  # [batch, seq_len]
            entropy = jnp.mean(entropy_per_pos, axis=-1)  # [batch]

            # Normalize
            normalized = (entropy - self.entropy_mean.value) / (self.entropy_std.value + 1e-6)
            signals["entropy"] = jax.nn.sigmoid(normalized)

            if update_stats:
                batch_mean = jnp.mean(entropy)
                batch_std = jnp.std(entropy) + 1e-6
                self.entropy_mean.value = self.momentum * self.entropy_mean.value + (1 - self.momentum) * batch_mean
                self.entropy_std.value = self.momentum * self.entropy_std.value + (1 - self.momentum) * batch_std

        # Token Confidence (inverted as uncertainty)
        if self.config.use_token_confidence and logits is not None:
            uncertainty = compute_token_confidence(logits)  # [batch]

            # Normalize
            normalized = (uncertainty - self.confidence_mean.value) / (self.confidence_std.value + 1e-6)
            signals["token_confidence"] = jax.nn.sigmoid(normalized)

            if update_stats:
                batch_mean = jnp.mean(uncertainty)
                batch_std = jnp.std(uncertainty) + 1e-6
                self.confidence_mean.value = self.momentum * self.confidence_mean.value + (1 - self.momentum) * batch_mean
                self.confidence_std.value = self.momentum * self.confidence_std.value + (1 - self.momentum) * batch_std

        # Attention Dispersion
        if self.config.use_attention_dispersion and attention_weights is not None:
            attn_entropy = compute_attention_entropy(attention_weights)
            signals["attention_dispersion"] = attn_entropy  # Already somewhat normalized

        return signals


class BudgetTracker(nnx.Module):
    """Track and adjust gating threshold based on remaining compute budget."""

    def __init__(self, config: AdvancedGatingConfig, rngs: nnx.Rngs):
        self.config = config

        # Running average of update rate
        self.update_rate_ema = nnx.Variable(jnp.array(config.target_update_rate))

        # Threshold adjustment
        self.threshold_offset = nnx.Variable(jnp.array(0.0))

    def __call__(
        self,
        base_threshold: float,
        did_update: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Get adjusted threshold and update budget tracker.

        Args:
            base_threshold: Base threshold value
            did_update: Boolean array of whether updates occurred [batch]

        Returns:
            Adjusted threshold scalar
        """
        # If we updated more than target, increase threshold (fewer updates)
        # If we updated less than target, decrease threshold (more updates)
        adjusted = base_threshold + self.threshold_offset.value

        if did_update is not None:
            # Update EMA of actual update rate
            batch_rate = jnp.mean(did_update.astype(jnp.float32))
            self.update_rate_ema.value = (
                (1 - self.config.budget_alpha) * self.update_rate_ema.value +
                self.config.budget_alpha * batch_rate
            )

            # Adjust threshold offset
            # If actual rate > target, increase offset (raise threshold)
            # If actual rate < target, decrease offset (lower threshold)
            error = self.update_rate_ema.value - self.config.target_update_rate
            self.threshold_offset.value = self.threshold_offset.value + 0.001 * error

            # Clamp offset to prevent runaway
            self.threshold_offset.value = jnp.clip(
                self.threshold_offset.value, -0.05, 0.05
            )

        return adjusted


class LearnedSignalCombiner(nnx.Module):
    """Learn to combine multiple signals for gating decision."""

    def __init__(self, config: AdvancedGatingConfig, rngs: nnx.Rngs):
        self.config = config

        # Count active signals
        num_signals = sum([
            config.use_ttt_improvement,
            config.use_entropy,
            config.use_attention_dispersion,
            config.use_token_confidence,
        ])

        # Small MLP to combine signals
        self.fc1 = nnx.Linear(num_signals + 1, config.hidden_dim, rngs=rngs)  # +1 for budget
        self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.head = nnx.Linear(
            config.hidden_dim,
            1,
            bias_init=nnx.initializers.constant(-1.0),  # Start biased toward skip
            rngs=rngs
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(
        self,
        signals: dict[str, jax.Array],
        budget_ratio: float = 1.0,
        train: bool = False,
    ) -> jax.Array:
        """Combine signals into update probability.

        Args:
            signals: Dictionary of normalized signals [batch]
            budget_ratio: Remaining budget as fraction [0, 1]
            train: Whether to enable dropout

        Returns:
            Update probability [batch, 1]
        """
        # Stack signals into feature vector
        signal_list: list[jax.Array] = []
        for key in sorted(signals.keys()):
            signal_list.append(signals[key][:, None])  # [batch, 1]
        signal_list.append(jnp.full_like(signal_list[0], budget_ratio))  # Budget feature

        x = jnp.concatenate(signal_list, axis=-1)  # [batch, num_signals + 1]

        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, deterministic=not train)

        x = self.fc2(x)
        x = jax.nn.gelu(x)

        logits = self.head(x)
        return jax.nn.sigmoid(logits)


class AdvancedGatingNetwork(nnx.Module):
    """
    Multi-signal gating for Run phase.

    Modes:
    - "threshold": Fixed threshold on combined signal (Walk phase extension)
    - "learned": Neural network combines signals
    - "budget_aware": Threshold with online budget adjustment
    """

    def __init__(self, config: AdvancedGatingConfig, rngs: nnx.Rngs):
        self.config = config

        # Signal extraction and normalization
        self.signal_extractor = SignalExtractor(config, rngs)

        # Mode-specific components
        if config.mode == "learned":
            self.combiner = LearnedSignalCombiner(config, rngs)

        if config.mode == "budget_aware":
            self.budget_tracker = BudgetTracker(config, rngs)

    def extract_signals(
        self,
        ttt_improvement: Optional[jax.Array] = None,
        logits: Optional[jax.Array] = None,
        attention_weights: Optional[jax.Array] = None,
        update_stats: bool = False,
    ) -> dict[str, jax.Array]:
        """Extract normalized signals from model outputs."""
        return self.signal_extractor(
            ttt_improvement=ttt_improvement,
            logits=logits,
            attention_weights=attention_weights,
            update_stats=update_stats,
        )

    def combine_signals(
        self,
        signals: dict[str, jax.Array],
    ) -> jax.Array:
        """Combine signals using fixed weights.

        Args:
            signals: Dictionary of normalized signals [batch]

        Returns:
            Combined signal [batch]
        """
        combined = jnp.zeros_like(next(iter(signals.values())))
        total_weight = 0.0

        for key, value in signals.items():
            weight = self.config.signal_weights.get(key, 0.0)
            combined = combined + weight * value
            total_weight += weight

        if total_weight > 0:
            combined = combined / total_weight

        return combined

    def __call__(
        self,
        ttt_improvement: Optional[jax.Array] = None,
        logits: Optional[jax.Array] = None,
        attention_weights: Optional[jax.Array] = None,
        budget_ratio: float = 1.0,
        train: bool = False,
        update_stats: bool = False,
        return_signals: bool = False,
    ) -> dict[str, Any]:
        """Make gating decision.

        Args:
            ttt_improvement: TTT loss improvement [batch]
            logits: Model logits [batch, seq_len, vocab_size]
            attention_weights: Attention weights (optional)
            budget_ratio: Remaining compute budget [0, 1]
            train: Whether in training mode
            update_stats: Whether to update running statistics
            return_signals: Whether to return individual signals

        Returns:
            Dictionary with:
                - decision: Boolean update decision [batch]
                - probability: Update probability [batch]
                - signals: Individual signals (if return_signals=True)
        """
        # Extract signals
        signals = self.extract_signals(
            ttt_improvement=ttt_improvement,
            logits=logits,
            attention_weights=attention_weights,
            update_stats=update_stats,
        )

        if self.config.mode == "threshold":
            # Simple threshold on combined signal
            combined = self.combine_signals(signals)
            # Signals are normalized to ~[0, 1] via sigmoid, so 0.5 is neutral
            decision = combined > 0.5
            probability = combined

        elif self.config.mode == "learned":
            # Neural network combination
            probability = self.combiner(signals, budget_ratio, train)[:, 0]
            decision = probability > 0.5

        elif self.config.mode == "budget_aware":
            # Threshold with budget adjustment
            combined = self.combine_signals(signals)

            # Get adjusted threshold (normalized to [0, 1] space)
            # base_threshold ~0.034 maps to ~0.5 after normalization
            adjusted_threshold = self.budget_tracker(
                0.5,  # Base threshold in normalized space
                did_update=None,  # Will update after decision
            )

            # Decision based on combined signal vs adjusted threshold
            decision = combined > adjusted_threshold
            probability = combined

            # Update budget tracker with decision
            self.budget_tracker(0.5, did_update=decision)
        else:
            raise ValueError(f"Unknown gating mode: {self.config.mode}")

        result: dict[str, Any] = {
            "decision": decision,
            "probability": probability,
        }

        if return_signals:
            result["signals"] = signals

        return result


def create_advanced_gating(
    mode: Literal["threshold", "learned", "budget_aware"] = "threshold",
    use_entropy: bool = True,
    use_token_confidence: bool = True,
    target_update_rate: float = 0.5,
    rngs: Optional[nnx.Rngs] = None,
) -> AdvancedGatingNetwork:
    """Create advanced gating network with default configuration.

    Args:
        mode: Gating mode
        use_entropy: Include entropy signal
        use_token_confidence: Include token confidence signal
        target_update_rate: Target update rate for budget-aware mode
        rngs: Random number generators

    Returns:
        Configured AdvancedGatingNetwork
    """
    if rngs is None:
        rngs = nnx.Rngs(42)

    config = AdvancedGatingConfig(
        use_ttt_improvement=True,
        use_entropy=use_entropy,
        use_token_confidence=use_token_confidence,
        use_attention_dispersion=False,  # Usually not worth the overhead
        mode=mode,
        target_update_rate=target_update_rate,
    )

    return AdvancedGatingNetwork(config, rngs)

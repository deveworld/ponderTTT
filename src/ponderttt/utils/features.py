"""
Feature extraction for policy network.
"""

import jax
import jax.numpy as jnp


class FeatureExtractor:
    """
    Extract 32-dimensional features for policy decisions.

    Features include:
    - Model confidence (4): entropy, perplexity
    - Activation stats (6): mean, std, sparsity
    - Attention patterns (4): entropy, range
    - Code metrics (8): token stats
    - Historical context (4): EMA, budget
    - Sequence stats (6): length, frequency

    Attributes:
        vocab_size: Vocabulary size
        ema_alpha: EMA weight for history tracking
    """

    def __init__(
        self,
        vocab_size: int,
        ema_alpha: float = 0.1,
        pad_token_id: int | None = None,
        seq_length_norm: float | None = None,
    ):
        self.vocab_size = vocab_size
        self.ema_alpha = ema_alpha
        self.pad_token_id = pad_token_id
        self.seq_length_norm = float(seq_length_norm) if seq_length_norm is not None else 4096.0

        # History tracking
        self.difficulty_ema = 0.0
        self.cost_ema = 0.0

    def extract(
        self,
        input_ids: jnp.ndarray,
        logits: jnp.ndarray,
        hidden_states: list[jnp.ndarray] | None = None,
        attentions: list[jnp.ndarray] | None = None,
        budget_remaining: float = 1.0,
    ) -> jnp.ndarray:
        """
        Extract features from model outputs.

        Args:
            input_ids: Input IDs [batch, seq_len]
            logits: Model logits [batch, seq_len, vocab_size]
            hidden_states: Hidden states from each layer
            attentions: Attention weights from each layer
            budget_remaining: Remaining budget fraction

        Returns:
            Features [batch, 32]
        """
        batch_size = input_ids.shape[0]

        features = []

        # 1. Model confidence (4)
        confidence = self._extract_confidence(logits)
        features.append(confidence)

        # 2. Activation statistics (6)
        if hidden_states is not None:
            activations = self._extract_activations(hidden_states)
        else:
            activations = jnp.zeros((batch_size, 6))
        features.append(activations)

        # 3. Attention patterns (4)
        if attentions is not None:
            attention_feat = self._extract_attention(attentions)
        else:
            attention_feat = jnp.zeros((batch_size, 4))
        features.append(attention_feat)

        # 4. Code metrics (8)
        code_feat = self._extract_code_metrics(input_ids, logits)
        features.append(code_feat)

        # 5. Historical context (4)
        history_feat = self._extract_history(batch_size, budget_remaining)
        features.append(history_feat)

        # 6. Sequence statistics (6)
        seq_feat = self._extract_sequence(input_ids)
        features.append(seq_feat)

        # Concatenate all features
        all_features = jnp.concatenate(features, axis=-1)

        assert all_features.shape[-1] == 32, (
            f"Expected 32 features, got {all_features.shape[-1]}"
        )

        return all_features

    def _extract_confidence(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Extract model confidence features (4D)."""
        # Compute probabilities
        probs = jax.nn.softmax(logits, axis=-1)

        # Entropy
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
        mean_entropy = jnp.mean(entropy, axis=-1)
        max_entropy = jnp.max(entropy, axis=-1)

        # Perplexity (use max prob as proxy)
        max_probs = jnp.max(probs, axis=-1)
        mean_perplexity = jnp.exp(-jnp.mean(jnp.log(max_probs + 1e-10), axis=-1))
        max_perplexity = jnp.exp(-jnp.min(jnp.log(max_probs + 1e-10), axis=-1))

        return jnp.stack(
            [
                mean_entropy,
                max_entropy,
                jnp.log(mean_perplexity + 1e-10),
                jnp.log(max_perplexity + 1e-10),
            ],
            axis=-1,
        )

    def _extract_activations(self, hidden_states: list[jnp.ndarray]) -> jnp.ndarray:
        """Extract activation statistics (6D)."""
        # Use last layer
        last_hidden = hidden_states[-1]  # [batch, seq_len, hidden_dim]

        # Statistics
        mean_act = jnp.mean(last_hidden, axis=(1, 2))
        std_act = jnp.std(last_hidden, axis=(1, 2))
        sparsity = jnp.mean(
            (jnp.abs(last_hidden) < 0.01).astype(jnp.float32), axis=(1, 2)
        )
        max_act = jnp.max(last_hidden, axis=(1, 2))
        min_act = jnp.min(last_hidden, axis=(1, 2))
        range_act = max_act - min_act

        return jnp.stack(
            [
                mean_act,
                std_act,
                sparsity,
                max_act,
                min_act,
                range_act,
            ],
            axis=-1,
        )

    def _extract_attention(self, attentions: list[jnp.ndarray]) -> jnp.ndarray:
        """Extract attention pattern features (4D)."""
        # Use last layer attention
        last_attn = attentions[-1]  # [batch, num_heads, seq_len, seq_len]

        # Average over heads
        attn_avg = jnp.mean(last_attn, axis=1)

        # Entropy
        entropy = -jnp.sum(attn_avg * jnp.log(attn_avg + 1e-10), axis=-1)
        mean_entropy = jnp.mean(entropy, axis=-1)
        max_entropy = jnp.max(entropy, axis=-1)

        # Range
        attn_range = jnp.mean(
            jnp.max(last_attn, axis=-1) - jnp.min(last_attn, axis=-1), axis=(1, 2)
        )

        # Sparsity
        sparsity = jnp.mean((last_attn < 0.01).astype(jnp.float32), axis=(1, 2, 3))

        return jnp.stack(
            [
                mean_entropy,
                max_entropy,
                attn_range,
                sparsity,
            ],
            axis=-1,
        )

    def _extract_code_metrics(
        self,
        input_ids: jnp.ndarray,
        logits: jnp.ndarray,
    ) -> jnp.ndarray:
        """Extract code-specific metrics (8D)."""
        batch_size, seq_len = input_ids.shape

        # Token statistics
        # Diversity metric (replaces unique token ratio for JIT compatibility)
        # Use standard deviation as a proxy for token diversity
        # Higher std = more diverse tokens
        token_diversity = (
            jnp.std(input_ids.astype(jnp.float32), axis=-1) / self.vocab_size
        )

        # Repetition (adjacent duplicates)
        repeated = jnp.mean(
            (input_ids[:, 1:] == input_ids[:, :-1]).astype(jnp.float32), axis=-1
        )

        # Token ID statistics
        avg_token_id = (
            jnp.mean(input_ids.astype(jnp.float32), axis=-1) / self.vocab_size
        )
        std_token_id = jnp.std(input_ids.astype(jnp.float32), axis=-1) / self.vocab_size

        # Prediction confidence
        probs = jax.nn.softmax(logits, axis=-1)
        max_probs = jnp.max(probs, axis=-1)
        pred_confidence = jnp.mean(max_probs, axis=-1)

        # Top-k diversity
        top_k_probs = jax.lax.top_k(probs, 5)[0]
        top_k_probs = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)
        top_k_entropy = -jnp.sum(top_k_probs * jnp.log(top_k_probs + 1e-10), axis=-1)
        top_k_diversity = jnp.mean(top_k_entropy, axis=-1)

        # Token variation
        token_diffs = jnp.abs(
            input_ids[:, 1:].astype(jnp.float32) - input_ids[:, :-1].astype(jnp.float32)
        )
        token_variation = jnp.std(token_diffs, axis=-1) / self.vocab_size

        # Prediction uncertainty (std of top-k probabilities)
        top_k_probs = jax.lax.top_k(probs, 5)[0]
        pred_uncertainty = jnp.std(top_k_probs, axis=-1)
        pred_uncertainty = jnp.mean(pred_uncertainty, axis=-1)

        return jnp.stack(
            [
                token_diversity,
                repeated,
                avg_token_id,
                std_token_id,
                pred_uncertainty,  # Fixed: was duplicate token_diversity
                pred_confidence,
                top_k_diversity,
                token_variation,
            ],
            axis=-1,
        )

    def _extract_history(
        self,
        batch_size: int,
        budget_remaining: float,
    ) -> jnp.ndarray:
        """Extract historical context (4D)."""
        difficulty = jnp.full((batch_size,), self.difficulty_ema)
        cost = jnp.full((batch_size,), self.cost_ema)
        budget_rem = jnp.full((batch_size,), budget_remaining)
        budget_util = jnp.full((batch_size,), 1.0 - budget_remaining)

        return jnp.stack([difficulty, cost, budget_rem, budget_util], axis=-1)

    def _extract_sequence(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Extract sequence statistics (6D)."""
        batch_size, seq_len = input_ids.shape

        # Optional mask based on pad token
        if self.pad_token_id is None:
            mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
        else:
            mask = (input_ids != self.pad_token_id).astype(jnp.float32)

        # Length (normalized)
        valid_tokens = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
        seq_length = valid_tokens / self.seq_length_norm

        # Token frequency (masked)
        avg_freq = jnp.sum(input_ids.astype(jnp.float32) * mask, axis=-1) / (valid_tokens * self.vocab_size)
        max_freq = jnp.max(jnp.where(mask > 0, input_ids, 0), axis=-1) / self.vocab_size

        # Position statistics (relative index of non-pad tokens)
        positions = jnp.arange(seq_len, dtype=jnp.float32)[None, :] / jnp.maximum(seq_len - 1, 1)
        position_mean = jnp.sum(positions * mask, axis=-1) / valid_tokens
        position_std = jnp.sqrt(
            jnp.sum(((positions - position_mean[:, None]) ** 2) * mask, axis=-1) / valid_tokens
        )

        # Compression ratio (proxy: fraction of non-pad tokens)
        compression = valid_tokens / seq_len

        return jnp.stack(
            [
                seq_length,
                jnp.log(avg_freq + 1.0),
                jnp.log(max_freq + 1.0),
                position_mean,
                position_std,
                compression,
            ],
            axis=-1,
        )

    def update_history(self, difficulty: float, cost: float):
        """Update EMA history."""
        self.difficulty_ema = (
            self.ema_alpha * difficulty + (1 - self.ema_alpha) * self.difficulty_ema
        )
        self.cost_ema = self.ema_alpha * cost + (1 - self.ema_alpha) * self.cost_ema

    def reset_history(self):
        """Reset history."""
        self.difficulty_ema = 0.0
        self.cost_ema = 0.0


def extract_features(
    input_ids: jnp.ndarray,
    logits: jnp.ndarray,
    vocab_size: int,
    pad_token_id: int | None = None,
    seq_length_norm: float | None = None,
    **kwargs,
) -> jnp.ndarray:
    """
    Standalone feature extraction function.

    Args:
        input_ids: Input IDs
        logits: Model logits
        vocab_size: Vocabulary size
        **kwargs: Additional arguments

    Returns:
        Features [batch, 32]
    """
    extractor = FeatureExtractor(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        seq_length_norm=seq_length_norm,
    )
    return extractor.extract(input_ids, logits, **kwargs)

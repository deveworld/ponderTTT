"""
Fast weight adaptation using LoRA-style low-rank updates.

Inspired by LaCT's fast weight mechanism.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
from dataclasses import field


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for a linear layer.

    Implements: y = Wx + (BA)x, where B and A are low-rank matrices.

    Attributes:
        features: Output dimension
        rank: Rank of adaptation
        alpha: Scaling factor
        dtype: Data type
    """
    features: int
    rank: int = 64
    alpha: float = 1.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply LoRA adaptation.

        Args:
            x: Input [batch, seq_len, in_features]

        Returns:
            Output [batch, seq_len, features]
        """
        in_features = x.shape[-1]

        # Low-rank matrices
        lora_A = self.param(
            'lora_A',
            nn.initializers.normal(stddev=0.02),
            (in_features, self.rank),
            self.dtype,
        )
        lora_B = self.param(
            'lora_B',
            nn.initializers.zeros,
            (self.rank, self.features),
            self.dtype,
        )

        # Compute low-rank update: x @ A @ B
        scaling = self.alpha / self.rank
        adaptation = jnp.dot(jnp.dot(x, lora_A), lora_B) * scaling

        return adaptation


class FastWeightModule(nn.Module):
    """
    Fast weight module for TTT.

    Implements SwiGLU-style fast weights inspired by LaCT:
        y = w1 @ (silu(w0 @ x) * (w2 @ x))

    Attributes:
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        dtype: Data type
    """
    hidden_dim: int
    output_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        w0: Optional[jnp.ndarray] = None,
        w1: Optional[jnp.ndarray] = None,
        w2: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Apply fast weight transformation.

        Args:
            x: Input [batch, seq_len, in_dim]
            w0, w1, w2: Fast weight matrices (if None, use parameters)

        Returns:
            Output [batch, seq_len, output_dim]
        """
        in_dim = x.shape[-1]

        # Initialize fast weights if not provided
        if w0 is None:
            w0 = self.param(
                'w0',
                nn.initializers.normal(stddev=0.02),
                (in_dim, self.hidden_dim),
                self.dtype,
            )
        if w1 is None:
            w1 = self.param(
                'w1',
                nn.initializers.normal(stddev=0.02),
                (self.hidden_dim, self.output_dim),
                self.dtype,
            )
        if w2 is None:
            w2 = self.param(
                'w2',
                nn.initializers.normal(stddev=0.02),
                (in_dim, self.hidden_dim),
                self.dtype,
            )

        # SwiGLU: w1 @ (silu(w0 @ x) * (w2 @ x))
        gate = nn.silu(jnp.dot(x, w0))
        hidden = jnp.dot(x, w2)
        activated = gate * hidden
        output = jnp.dot(activated, w1)

        return output


def compute_fast_weight_update(
    grad: jnp.ndarray,
    weight: jnp.ndarray,
    learning_rate: float,
) -> jnp.ndarray:
    """
    Compute fast weight update with gradient descent.

    Args:
        grad: Gradient wrt weight
        weight: Current weight
        learning_rate: Learning rate

    Returns:
        Updated weight
    """
    return weight - learning_rate * grad


def compute_normalized_gradient(grad: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize gradient for stability (L2 normalization).

    Args:
        grad: Raw gradient

    Returns:
        Normalized gradient
    """
    norm = jnp.linalg.norm(grad, axis=-1, keepdims=True)
    return grad / (norm + 1e-5)


def newton_schulz_orthogonalize(
    G: jnp.ndarray,
    num_iters: int = 5,
) -> jnp.ndarray:
    """
    Newton-Schulz iteration for matrix orthogonalization.

    From LaCT's zeropower_via_newtonschulz5.

    Args:
        G: Input matrix [batch, d1, d2]
        num_iters: Number of iterations

    Returns:
        Orthogonalized matrix
    """
    X = G.astype(jnp.float32)

    # Transpose if needed
    transpose = G.shape[-2] > G.shape[-1]
    if transpose:
        X = jnp.swapaxes(X, -2, -1)

    # Ensure spectral norm <= 1
    norm = jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)
    X = X / (norm + 1e-7)

    # Newton-Schulz iterations
    coeffs = [
        (3.0848, -4.6946, 1.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    for a, b, c in coeffs[:num_iters]:
        A = X @ jnp.swapaxes(X, -2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transpose:
        X = jnp.swapaxes(X, -2, -1)

    return X.astype(G.dtype)

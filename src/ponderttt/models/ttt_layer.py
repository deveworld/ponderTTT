"""
Test-Time Training Layer implementation in Flax.

Based on TTT-LM-JAX architecture.
"""

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from .fast_weights import FastWeightModule, compute_fast_weight_update


@dataclass
class TTTConfig:
    """Configuration for TTT layer."""

    hidden_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    ttt_hidden_dim: int = 2048
    chunk_size: int = 512
    max_seq_length: int = 8192
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32


class TTTLayer(nn.Module):
    """
    Test-Time Training Layer.

    Performs self-supervised learning at test time using fast weights.

    Attributes:
        config: TTT configuration
    """

    config: TTTConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        deterministic: bool = True,
        enable_internal_updates: bool = False,
    ) -> tuple[jnp.ndarray, dict]:
        """
        Apply TTT layer.

        Args:
            x: Input [batch, seq_len, hidden_dim]
            mask: Attention mask [batch, seq_len]
            deterministic: Whether to use dropout
            enable_internal_updates: If True, do self-supervised internal updates (for PonderTTT).
                                    If False, simple feedforward pass (for baselines).

        Returns:
            output: Transformed sequence [batch, seq_len, hidden_dim]
            ttt_stats: Dictionary with TTT statistics
        """
        batch_size, seq_len, hidden_dim = x.shape
        cfg = self.config

        # Query, Key, Value projections
        qkv = nn.Dense(
            features=3 * hidden_dim, use_bias=False, dtype=cfg.dtype, name="qkv_proj"
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        k = k.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        v = v.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)

        # Initialize fast weights
        k_flat_dim = cfg.num_heads * cfg.head_dim
        w0 = self.param(
            "fast_w0",
            nn.initializers.normal(stddev=0.02),
            (k_flat_dim, cfg.ttt_hidden_dim),
            cfg.dtype,
        )
        w1 = self.param(
            "fast_w1",
            nn.initializers.normal(stddev=0.02),
            (cfg.ttt_hidden_dim, hidden_dim),
            cfg.dtype,
        )
        w2 = self.param(
            "fast_w2",
            nn.initializers.normal(stddev=0.02),
            (k_flat_dim, cfg.ttt_hidden_dim),
            cfg.dtype,
        )

        #  Branch: Simple feedforward OR internal TTT updates
        if not enable_internal_updates:
            # BASELINE MODE: Simple feedforward without internal updates
            # Just apply fast weight transformation once
            k_flat = k.reshape(batch_size, seq_len, k_flat_dim)

            # Normalize input to stable range
            k_flat_normalized = k_flat / jnp.sqrt(jnp.float32(k_flat_dim))

            # SwiGLU-style gating (similar to official TTT)
            gate = nn.silu(jnp.dot(k_flat_normalized, w0))
            hidden = jnp.dot(k_flat_normalized, w2)
            activated = gate * hidden

            # Normalize before final projection
            activated = activated / jnp.sqrt(jnp.float32(cfg.ttt_hidden_dim))
            v_transformed = jnp.dot(activated, w1)
            v_transformed = v_transformed.reshape(
                batch_size, seq_len, cfg.num_heads, cfg.head_dim
            )

            # Apply query attention-like mechanism
            # output: [batch, seq, num_heads, head_dim]
            ttt_output = q * v_transformed
            # Flatten multi-head back to hidden_dim: [batch, seq, hidden_dim]
            ttt_output = ttt_output.reshape(batch_size, seq_len, hidden_dim)

            # CRITICAL: Add skip connection (residual) like in transformers
            # This ensures output is in the same range as input
            output = x + ttt_output

            ttt_stats = {"ttt_loss": 0.0, "num_chunks": 0}

        else:
            # PONDERTTT MODE: Self-supervised internal TTT updates
            # Process in chunks for TTT using scan (JIT-compatible)
            num_chunks = seq_len // cfg.chunk_size
            learning_rate = 1e-3

            def process_chunk(carry, chunk_idx):
                """Process a single chunk with gradient-based weight updates."""
                current_w0, current_w1, current_w2 = carry

                start_idx = chunk_idx * cfg.chunk_size

                # Extract chunk using dynamic_slice (JIT-compatible)
                k_chunk = jax.lax.dynamic_slice(
                    k,
                    (0, start_idx, 0, 0),
                    (batch_size, cfg.chunk_size, cfg.num_heads, cfg.head_dim),
                )
                v_chunk = jax.lax.dynamic_slice(
                    v,
                    (0, start_idx, 0, 0),
                    (batch_size, cfg.chunk_size, cfg.num_heads, cfg.head_dim),
                )

                # Perform TTT update with gradient descent
                chunk_output, chunk_loss, updated_w0, updated_w1, updated_w2 = (
                    self._ttt_update_chunk(
                        k_chunk,
                        v_chunk,
                        current_w0,
                        current_w1,
                        current_w2,
                        learning_rate,
                    )
                )

                new_carry = (updated_w0, updated_w1, updated_w2)
                return new_carry, (chunk_output, chunk_loss)

            # Use scan to process all chunks with weight updates
            init_carry = (w0, w1, w2)
            _, (chunk_outputs, chunk_losses) = jax.lax.scan(
                process_chunk,
                init_carry,
                jnp.arange(num_chunks),
            )

            # Concatenate chunks
            ttt_output = jnp.concatenate(chunk_outputs, axis=1)
            ttt_stats = {
                "ttt_loss": float(jnp.mean(chunk_losses)),
                "num_chunks": int(num_chunks),
            }

            # Apply query to get final output and flatten to hidden_dim
            output_reshaped = ttt_output.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
            # output: [batch, seq, num_heads, head_dim]
            ttt_output = q * output_reshaped
            # Flatten multi-head back to hidden_dim: [batch, seq, hidden_dim]
            ttt_output = ttt_output.reshape(batch_size, seq_len, hidden_dim)

            # CRITICAL: Add skip connection (residual) like in transformers
            output = x + ttt_output

        # Output projection
        output = nn.Dense(
            features=hidden_dim, use_bias=False, dtype=cfg.dtype, name="output_proj"
        )(output)

        # Dropout
        if not deterministic:
            output = nn.Dropout(rate=cfg.dropout_rate)(output, deterministic=False)

        return output, ttt_stats

    @nn.compact
    def apply_simple_forward(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Simple feedforward pass WITHOUT internal TTT updates.

        For baseline evaluation: TTT layer is just a learnable transformation.
        No self-supervised updates - only external gradient descent applies.

        Args:
            x: Input [batch, seq_len, hidden_dim]
            deterministic: Whether to use dropout

        Returns:
            output: Transformed sequence [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        cfg = self.config

        # QKV projections
        qkv = nn.Dense(
            features=3 * hidden_dim, use_bias=False, dtype=cfg.dtype, name="qkv_proj"
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        k = k.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        v = v.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)

        # Get fast weight parameters (but don't update them internally)
        k_flat_dim = cfg.num_heads * cfg.head_dim
        w0 = self.param(
            "fast_w0",
            nn.initializers.normal(stddev=0.02),
            (k_flat_dim, cfg.ttt_hidden_dim),
            cfg.dtype,
        )
        w1 = self.param(
            "fast_w1",
            nn.initializers.normal(stddev=0.02),
            (cfg.ttt_hidden_dim, hidden_dim),
            cfg.dtype,
        )
        w2 = self.param(
            "fast_w2",
            nn.initializers.normal(stddev=0.02),
            (k_flat_dim, cfg.ttt_hidden_dim),
            cfg.dtype,
        )

        # Simple feedforward transformation (no internal updates)
        k_flat = k.reshape(batch_size, seq_len, k_flat_dim)

        # Normalize input to stable range
        k_flat_normalized = k_flat / jnp.sqrt(jnp.float32(k_flat_dim))

        # SwiGLU-style gating
        gate = nn.silu(jnp.dot(k_flat_normalized, w0))
        hidden = jnp.dot(k_flat_normalized, w2)
        activated = gate * hidden

        # Normalize before final projection
        activated = activated / jnp.sqrt(jnp.float32(cfg.ttt_hidden_dim))
        v_transformed = jnp.dot(activated, w1)
        v_transformed = v_transformed.reshape(
            batch_size, seq_len, cfg.num_heads, cfg.head_dim
        )

        # Apply query attention-like mechanism
        # output: [batch, seq, num_heads, head_dim]
        ttt_output = q * v_transformed
        # Flatten multi-head back to hidden_dim: [batch, seq, hidden_dim]
        ttt_output = ttt_output.reshape(batch_size, seq_len, hidden_dim)

        # Output projection
        ttt_output = nn.Dense(
            features=hidden_dim, use_bias=False, dtype=cfg.dtype, name="output_proj"
        )(ttt_output)

        # CRITICAL: Add skip connection (residual)
        output = x + ttt_output

        # Dropout
        if not deterministic:
            output = nn.Dropout(rate=cfg.dropout_rate)(output, deterministic=False)

        return output

    def _ttt_update_chunk(
        self,
        k: jnp.ndarray,
        v: jnp.ndarray,
        w0: jnp.ndarray,
        w1: jnp.ndarray,
        w2: jnp.ndarray,
        learning_rate: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Perform TTT update on a single chunk with gradient descent.

        Args:
            k: Keys [batch, chunk_size, num_heads, head_dim]
            v: Values [batch, chunk_size, num_heads, head_dim]
            w0: Fast weight matrix 0 [k_dim, hidden_dim]
            w1: Fast weight matrix 1 [hidden_dim, v_dim]
            w2: Fast weight matrix 2 [k_dim, hidden_dim]
            learning_rate: Learning rate for fast weight updates

        Returns:
            output: Updated values [batch, chunk_size, num_heads, head_dim]
            loss: TTT loss (self-supervised)
            updated_w0: Updated weight matrix 0
            updated_w1: Updated weight matrix 1
            updated_w2: Updated weight matrix 2
        """
        batch_size, chunk_size, num_heads, head_dim = k.shape

        # Flatten for processing
        k_flat = k.reshape(batch_size, chunk_size, -1)
        v_flat = v.reshape(batch_size, chunk_size, -1)

        # Define forward pass with fast weights (SwiGLU style)
        def forward_fn(w0_param, w1_param, w2_param, k_input):
            """Forward pass through fast weight network."""
            gate = nn.silu(jnp.dot(k_input, w0_param))
            hidden = jnp.dot(k_input, w2_param)
            activated = gate * hidden
            output = jnp.dot(activated, w1_param)
            return output

        # Define loss function for gradient computation
        def loss_fn(w0_param, w1_param, w2_param):
            """Compute TTT loss: predict values from keys."""
            v_pred = forward_fn(w0_param, w1_param, w2_param, k_flat)
            mse_loss = jnp.mean((v_pred - v_flat) ** 2)
            return mse_loss

        # Compute gradients efficiently using single backward pass
        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(w0, w1, w2)
        grad_w0, grad_w1, grad_w2 = grads

        # Update weights using gradient descent
        updated_w0 = compute_fast_weight_update(grad_w0, w0, learning_rate)
        updated_w1 = compute_fast_weight_update(grad_w1, w1, learning_rate)
        updated_w2 = compute_fast_weight_update(grad_w2, w2, learning_rate)

        # Generate output with updated weights
        output = forward_fn(updated_w0, updated_w1, updated_w2, k_flat)

        # Reshape output back to multi-head format
        output = output.reshape(batch_size, chunk_size, num_heads, head_dim)

        return output, loss, updated_w0, updated_w1, updated_w2


class ChunkedTTTLayer(nn.Module):
    """
    Chunked TTT Layer with explicit fast weight updates.

    More aligned with LaCT's chunked processing.

    Attributes:
        config: TTT configuration
        learning_rate: Fast weight learning rate
    """

    config: TTTConfig
    learning_rate: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict]:
        """
        Apply chunked TTT with fast weight updates.

        Args:
            x: Input [batch, seq_len, hidden_dim]
            mask: Attention mask

        Returns:
            output: Processed sequence
            stats: TTT statistics
        """
        cfg = self.config
        batch_size, seq_len, hidden_dim = x.shape

        # Initialize fast weights
        w0 = self.param(
            "fast_w0",
            nn.initializers.normal(stddev=0.02),
            (hidden_dim, cfg.ttt_hidden_dim),
            cfg.dtype,
        )
        w1 = self.param(
            "fast_w1",
            nn.initializers.normal(stddev=0.02),
            (cfg.ttt_hidden_dim, hidden_dim),
            cfg.dtype,
        )
        w2 = self.param(
            "fast_w2",
            nn.initializers.normal(stddev=0.02),
            (hidden_dim, cfg.ttt_hidden_dim),
            cfg.dtype,
        )

        # Process chunks using scan (JIT-compatible)
        num_chunks = seq_len // cfg.chunk_size

        # Fast weight module (instantiate once)
        fast_weight_fn = FastWeightModule(
            hidden_dim=cfg.ttt_hidden_dim,
            output_dim=hidden_dim,
            dtype=cfg.dtype,
        )

        def process_chunk(carry, chunk_idx):
            """Process a single chunk with fast weight updates."""
            current_w0, current_w1, current_w2, accumulated_loss = carry

            start_idx = chunk_idx * cfg.chunk_size

            # Extract chunk using dynamic_slice (JIT-compatible)
            x_chunk = jax.lax.dynamic_slice(
                x, (0, start_idx, 0), (batch_size, cfg.chunk_size, hidden_dim)
            )

            # Forward pass with fast weights
            chunk_output = fast_weight_fn(
                x_chunk,
                w0=current_w0,
                w1=current_w1,
                w2=current_w2,
            )

            # Self-supervised loss (reconstruct input)
            def loss_fn(w0, w1, w2):
                output = fast_weight_fn(x_chunk, w0=w0, w1=w1, w2=w2)
                return jnp.mean((output - x_chunk) ** 2)

            chunk_loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(
                current_w0, current_w1, current_w2
            )
            grad_w0, grad_w1, grad_w2 = grads
            accumulated_loss = accumulated_loss + chunk_loss

            # Update fast weights using gradient descent
            updated_w0 = current_w0 - self.learning_rate * grad_w0
            updated_w1 = current_w1 - self.learning_rate * grad_w1
            updated_w2 = current_w2 - self.learning_rate * grad_w2

            new_carry = (updated_w0, updated_w1, updated_w2, accumulated_loss)
            return new_carry, chunk_output

        # Initial carry: weights and accumulated loss
        init_carry = (w0, w1, w2, 0.0)

        # Use scan to process all chunks
        (final_w0, final_w1, final_w2, total_loss), chunk_outputs = jax.lax.scan(
            process_chunk,
            init_carry,
            jnp.arange(num_chunks),
        )

        output = jnp.concatenate(chunk_outputs, axis=1)
        stats = {
            "ttt_loss": total_loss / num_chunks,
            "num_chunks": num_chunks,
        }

        return output, stats

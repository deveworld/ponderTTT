"""
Test-Time Training Layer implementation in Flax NNX.

Migrated from Linen to NNX while preserving TTT algorithm exactly.
Based on official TTT-LM-JAX implementation.
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import vmap
from jax.tree_util import tree_map


def scan_remat_every_n_iterations_scan(f, n, carry, x):
    """
    Remat every n mini batches for memory efficiency.

    From official TTT implementation.
    Groups mini-batches into chunks of size n, applies jax.remat to reduce
    memory usage during backpropagation.
    """
    x_grouped = tree_map(lambda x: x.reshape((-1, n, *x.shape[1:])), x)
    carry, y_grouped = jax.lax.scan(
        jax.remat(partial(jax.lax.scan, f), prevent_cse=False),
        carry,
        x_grouped
    )
    y = tree_map(lambda x: x.reshape((-1, *x.shape[2:])), y_grouped)
    return carry, y


@dataclass
class TTTConfig:
    """Configuration for TTT layer."""

    hidden_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    ttt_hidden_dim: int = 2048  # Not used in Linear variant
    mini_batch_size: int = 16  # Official TTT uses 16
    chunk_size: int = 512  # For compatibility
    max_seq_length: int = 8192
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32
    initializer_range: float = 0.02
    ttt_base_lr: float = 1.0
    rope_theta: float = 10000.0
    conv_width: int = 4
    remat_mini_batch_group_size: int = 4
    output_ttt_stats: bool = True


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """Precompute Rotary Position Embedding frequencies."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)
    freqs = np.outer(t, freqs).astype(dtype)
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray, dtype: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Rotary Position Embedding to queries and keys."""
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


class TTTLayerNorm(nnx.Module):
    """Per-head layer normalization for TTT."""

    def __init__(self, num_heads: int, head_dim: int, epsilon: float, rngs: nnx.Rngs):
        """Initialize per-head layer norm.

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension per head
            epsilon: Layer norm epsilon
            rngs: Random number generators
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.epsilon = epsilon

        # Per-head parameters
        self.scale = nnx.Param(jnp.ones((num_heads, 1, head_dim)))
        self.bias = nnx.Param(jnp.zeros((num_heads, 1, head_dim)))

    def __call__(self, x: jax.Array, head_idx: int) -> jax.Array:
        """Apply layer norm for specific head.

        Args:
            x: Input [mini_batch_size, head_dim]
            head_idx: Which head to use parameters for

        Returns:
            Normalized output [mini_batch_size, head_dim]
        """
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(var + self.epsilon)
        return normalized * self.scale.value[head_idx] + self.bias.value[head_idx]


class TTTLayer(nnx.Module):
    """
    Test-Time Training Layer in NNX.

    Based on official TTT-LM-JAX implementation (TTTLinear variant).
    Migrated from Linen to NNX while preserving exact algorithm.
    """

    def __init__(self, config: TTTConfig, rngs: nnx.Rngs):
        """Initialize TTT layer.

        Args:
            config: TTT configuration
            rngs: Random number generators
        """
        self.config = config
        self.width = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.mini_batch_size = config.mini_batch_size

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            self.mini_batch_size * 2,
            theta=config.rope_theta,
            dtype=config.dtype
        )

        # Q, K, V, O projections
        self.wq = nnx.Linear(config.hidden_dim, config.num_heads * config.head_dim, use_bias=False, rngs=rngs)
        self.wv = nnx.Linear(config.hidden_dim, config.num_heads * config.head_dim, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(config.num_heads * config.head_dim, config.hidden_dim, use_bias=False, rngs=rngs)
        self.wg = nnx.Linear(config.hidden_dim, config.hidden_dim, use_bias=False, rngs=rngs)

        # Causal convolutions for Q and K
        # Note: NNX Conv doesn't directly support CAUSAL padding like Linen
        # We'll handle causal masking manually in forward pass
        self.conv_q = nnx.Conv(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            kernel_size=(config.conv_width,),
            feature_group_count=config.hidden_dim,  # Depthwise
            rngs=rngs
        )
        self.conv_k = nnx.Conv(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            kernel_size=(config.conv_width,),
            feature_group_count=config.hidden_dim,  # Depthwise
            rngs=rngs
        )

        # Token position indices for learning rate
        self.token_idx = 1.0 / jnp.arange(1, self.mini_batch_size + 1, dtype=jnp.float32)
        self.learnable_token_idx = nnx.Param(jnp.zeros((self.mini_batch_size,), dtype=jnp.float32))

        # Learnable learning rate (per-head)
        # Create num_heads separate linear layers for per-head learning rate
        self.learnable_ttt_lr_layers = nnx.List([
            nnx.Linear(config.hidden_dim, 1, rngs=rngs) for _ in range(config.num_heads)
        ])

        # Per-head TTT normalization
        self.ttt_norm = TTTLayerNorm(config.num_heads, config.head_dim, 1e-5, rngs)

        # Post-normalization
        self.post_norm = nnx.LayerNorm(config.num_heads * config.head_dim, epsilon=1e-5, rngs=rngs)

        # Fast weights for TTT (per-head)
        normal_init = nnx.initializers.normal(config.initializer_range)
        self.W1 = nnx.Param(normal_init(rngs.params(), (config.num_heads, config.head_dim, config.head_dim)))
        self.b1 = nnx.Param(jnp.zeros((config.num_heads, 1, config.head_dim)))

    def _apply_causal_conv(self, conv_layer: nnx.Conv, x: jax.Array) -> jax.Array:
        """Apply causal convolution by padding on the left.

        Args:
            conv_layer: Convolution layer
            x: Input [batch, seq_len, features]

        Returns:
            Convolved output [batch, seq_len, features]
        """
        # Pad left with kernel_width - 1 zeros for causal conv
        kernel_width = self.config.conv_width
        original_seq_len = x.shape[1]
        x_padded = jnp.pad(x, ((0, 0), (kernel_width - 1, 0), (0, 0)), mode='constant')
        # Apply conv (NNX conv expects [batch, length, features])
        out = conv_layer(x_padded)
        # Trim to original sequence length
        return out[:, :original_seq_len, :]

    def _split_heads(self, hidden_states: jax.Array) -> jax.Array:
        """Reshape to multi-head format."""
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _split_mini_batches(self, hidden_states: jax.Array) -> jax.Array:
        """Split sequence into mini-batches."""
        B, N, num_head, head_dim = hidden_states.shape
        n_mini_batch = N // self.mini_batch_size
        seq_shape = (n_mini_batch, self.mini_batch_size)
        hidden_states = hidden_states.reshape(B, *seq_shape, self.num_heads, self.head_dim).transpose(
            0, 3, 1, 2, 4
        )
        return hidden_states

    def get_qkv_projections(self, batch: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Get Q, K, V projections with causal convolution."""
        xqk = self.wq(batch)
        XV = self.wv(batch)
        XQ = self._apply_causal_conv(self.conv_q, xqk)
        XK = self._apply_causal_conv(self.conv_k, xqk)
        return XQ, XK, XV

    def get_eta(self, X: jax.Array) -> jax.Array:
        """Compute learnable learning rate (per-head, position-dependent)."""
        # Per-head learnable learning rate
        # X shape: [B, n_mini_batch, mini_batch_size, width]
        learnable_lrs = []
        for head_idx, lr_layer in enumerate(self.learnable_ttt_lr_layers):
            lr = lr_layer(X)  # [B, n_mini_batch, mini_batch_size, 1]
            learnable_lrs.append(lr)

        learnable_ttt_lr = jnp.stack(learnable_lrs, axis=1)  # [B, num_heads, n_mini_batch, mini_batch_size, 1]
        learnable_ttt_lr = nnx.sigmoid(learnable_ttt_lr)

        # Position-dependent base learning rate
        token_idx = self.learnable_token_idx.value + self.token_idx
        token_idx = jnp.clip(token_idx, a_min=0.0)

        # Combined learning rate
        eta = (
            (self.config.ttt_base_lr * token_idx).reshape(1, 1, 1, token_idx.shape[0], -1)
            * learnable_ttt_lr
            / self.head_dim
        )
        return eta

    def get_ttt_inputs(
        self, batch: jax.Array, position_ids: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Tuple]:
        """Prepare inputs for TTT processing."""
        B, N, F = batch.shape
        n_mini_batch = N // self.mini_batch_size
        seq_shape = (n_mini_batch, self.mini_batch_size)
        X = batch.reshape(B, *seq_shape, self.width)

        # Get Q, K, V projections
        XQ, XK, XV = self.get_qkv_projections(batch)

        # Compute TTT statistics if requested (before split_heads, matching original)
        if self.config.output_ttt_stats:
            # Need to split heads for stats calculation
            XV_heads = self._split_heads(XV)
            XK_heads = self._split_heads(XK)

            # Sample last token from each mini-batch
            last_indices = jnp.arange(
                self.mini_batch_size - 1,
                XV_heads.shape[1],
                self.mini_batch_size,
            )
            XV_last_in_mini_batch = jnp.take(XV_heads, last_indices, axis=1)
            XK_last_in_mini_batch = jnp.take(XK_heads, last_indices, axis=1)

            ssl_tgt_last_in_mini_batch = XV_last_in_mini_batch - XK_last_in_mini_batch
            ssl_tgt_mean = (XV_heads - XK_heads).mean(axis=1, keepdims=True).reshape(B, 1, self.num_heads, self.head_dim)
            ssl_tgt_last_in_mini_batch_from_mean_mse = ((ssl_tgt_last_in_mini_batch - ssl_tgt_mean) ** 2).mean(
                axis=(0, 2, 3)
            )
        else:
            ssl_tgt_last_in_mini_batch_from_mean_mse = None

        # Reshape to multi-head
        XQ = self._split_heads(XQ)
        XK = self._split_heads(XK)
        XV = self._split_heads(XV)

        # Apply RoPE
        freqs_cis = jnp.take(self.freqs_cis, position_ids % self.mini_batch_size, axis=0)
        XQ, XK = apply_rotary_emb(XQ, XK, freqs_cis=freqs_cis, dtype=self.config.dtype)

        # Split into mini-batches
        XQ = self._split_mini_batches(XQ)
        XK = self._split_mini_batches(XK)
        XV = self._split_mini_batches(XV)

        # Get learning rate
        eta = self.get_eta(X)

        return (XQ, XK, XV, eta, (ssl_tgt_last_in_mini_batch_from_mean_mse,))

    def process_mini_batch(
        self,
        XQ_mini_batch: jax.Array,
        XK_mini_batch: jax.Array,
        XV_mini_batch: jax.Array,
        eta_mini_batch: jax.Array,
        ttt_params_init: Tuple[jax.Array, jax.Array],
        ttt_params_mini_batch_init: Tuple[jax.Array, jax.Array],
        head_idx: int,
    ) -> Tuple[Tuple[jax.Array, jax.Array], Tuple]:
        """
        Process a single mini-batch with TTT.

        Core TTT algorithm from official implementation.
        """
        W1_init, b1_init = ttt_params_mini_batch_init
        square_eta_mini_batch = eta_mini_batch[: self.mini_batch_size]
        last_eta_in_mini_batch = eta_mini_batch[-1][:, None]

        X1 = XK_mini_batch

        # Forward pass
        Z1 = X1 @ W1_init + b1_init

        # Apply TTT norm using head-specific parameters
        def ttt_norm_fn(z):
            return self.ttt_norm(z, head_idx)

        ttt_norm_out, ttt_norm_vjp = jax.vjp(ttt_norm_fn, Z1)

        # Self-supervised target: XV - XK (CRITICAL!)
        ssl_target = XV_mini_batch - XK_mini_batch

        # Gradient computation
        grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target
        grad_l_wrt_Z1 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0]

        # Calculate TTT loss
        if self.config.output_ttt_stats:
            ttt_loss_mse_step_0 = (grad_l_wrt_ttt_norm_out[-1] ** 2).mean()
            W1_0, b1_0 = ttt_params_init
            Z1_0 = X1 @ W1_0 + b1_0
            ttt_norm_out_0 = ttt_norm_fn(Z1_0)
            ttt_loss_mse_init = ((ttt_norm_out_0 - ssl_target)[-1] ** 2).mean()
        else:
            ttt_loss_mse_step_0 = None
            ttt_loss_mse_init = None

        # Causal processing with updated weights
        X1_bar = XQ_mini_batch
        Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0))  # Causal mask!
        b1_bar = b1_init - (square_eta_mini_batch * jnp.tril(jnp.ones_like(Attn1))) @ grad_l_wrt_Z1
        Z1_bar = X1_bar @ W1_init - (square_eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
        ttt_norm_out_bar = ttt_norm_fn(Z1_bar)

        # Output with residual connection
        output_mini_batch = X1_bar + ttt_norm_out_bar

        # Weight update for next mini-batch
        W1_bar_last = W1_init - (last_eta_in_mini_batch * X1).transpose(1, 0) @ grad_l_wrt_Z1
        b1_bar_last = b1_init - jnp.sum(last_eta_in_mini_batch * grad_l_wrt_Z1, axis=0, keepdims=True)

        # Calculate ttt loss with updated weights
        if self.config.output_ttt_stats:
            X1_last_fwd_new = X1[-1:] @ W1_bar_last + b1_bar_last
            X1_last_fwd_new = ttt_norm_fn(X1_last_fwd_new)
            ttt_loss_mse_step_1 = ((X1_last_fwd_new - ssl_target[-1:]) ** 2).mean()
        else:
            ttt_loss_mse_step_1 = None

        ttt_params_mini_batch_new = (W1_bar_last, b1_bar_last)

        return (
            ttt_params_mini_batch_new,
            (output_mini_batch, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1),
        )

    def ttt(
        self, XQ: jax.Array, XK: jax.Array, XV: jax.Array, eta: jax.Array
    ) -> Tuple[jax.Array, Tuple]:
        """Perform Test-Time Training across all mini-batches."""
        B, N = XV.shape[0], XV.shape[2] * XV.shape[3]

        @partial(vmap, axis_name="batch")
        def update_embed(XQ, XK, XV, eta):
            @partial(vmap, axis_name="head")
            def parallelize_over_heads(XQ, XK, XV, eta, W1, b1, head_idx):
                ttt_params_init = (W1, b1)

                def compute_mini_batch(ttt_params_mini_batch_init, inputs):
                    return self.process_mini_batch(
                        inputs["XQ"],
                        inputs["XK"],
                        inputs["XV"],
                        inputs["eta"],
                        ttt_params_init,
                        ttt_params_mini_batch_init,
                        head_idx,
                    )

                inputs = {"XQ": XQ, "XK": XK, "XV": XV, "eta": eta}

                # Use scan with remat for memory efficiency
                _, outputs = scan_remat_every_n_iterations_scan(
                    compute_mini_batch,
                    self.config.remat_mini_batch_group_size,
                    ttt_params_init,
                    inputs
                )
                Z, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1 = outputs
                return (Z.reshape(-1, self.head_dim), ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1)

            # Create head indices
            head_indices = jnp.arange(self.num_heads)
            outputs = parallelize_over_heads(
                XQ, XK, XV, eta, self.W1.value, self.b1.value, head_indices
            )
            return outputs

        outputs = update_embed(XQ, XK, XV, eta)
        Z, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1 = outputs
        Z = Z.transpose(0, 2, 1, 3).reshape(B, N, -1)

        if self.config.output_ttt_stats:
            ttt_loss_mse_init = ttt_loss_mse_init.mean(axis=(0, 1))
            ttt_loss_mse_step_0 = ttt_loss_mse_step_0.mean(axis=(0, 1))
            ttt_loss_mse_step_1 = ttt_loss_mse_step_1.mean(axis=(0, 1))

        return Z, (ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1)

    def __call__(
        self,
        hidden_states: jax.Array,
        mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, dict]:
        """Apply TTT layer.

        Args:
            hidden_states: Input [batch, seq_len, hidden_dim]
            mask: Attention mask (not used)
            position_ids: Position indices for RoPE

        Returns:
            (output, ttt_stats) tuple

        Note:
            TTT layer doesn't use dropout, so no train/eval mode distinction needed.
        """
        # Generate position IDs if not provided
        if position_ids is None:
            batch_size, seq_len = hidden_states.shape[:2]
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

        # Get TTT inputs
        XQ, XK, XV, eta, precompute_stats = self.get_ttt_inputs(hidden_states, position_ids=position_ids)

        # Perform TTT
        Z, ttt_stats = self.ttt(XQ, XK, XV, eta)

        # Post-normalization
        Z = self.post_norm(Z)

        # Apply gating
        y = self.wg(hidden_states)
        y = nnx.gelu(y)
        Z = y * Z

        # Output projection
        ttt_output = self.wo(Z)

        # Combine stats
        all_stats = {
            "ttt_loss_init": ttt_stats[0],
            "ttt_loss_step_0": ttt_stats[1],
            "ttt_loss_step_1": ttt_stats[2],
            "ssl_target_variance": precompute_stats[0],
        }

        return ttt_output, all_stats

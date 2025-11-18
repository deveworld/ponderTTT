"""
Test-Time Training Layer implementation in Flax.

Based on official TTT-LM-JAX implementation.
Adapted for PonderTTT with policy-guided adaptive compute.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple, Union, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.tree_util import tree_map


Axes = Union[int, Sequence[int]]


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
    ttt_hidden_dim: int = 2048  # Not used in Linear variant, kept for compatibility
    mini_batch_size: int = 16  # Official TTT uses 16
    chunk_size: int = 512  # For compatibility, but processing uses mini_batch_size
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
    """
    Precompute Rotary Position Embedding frequencies.

    From official TTT implementation.
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)
    freqs = np.outer(t, freqs).astype(dtype)
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray, dtype: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply Rotary Position Embedding to queries and keys.

    From official TTT implementation.
    """
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


def get_multi_head_params(self, params, param_dtype, kernel_init="normal", std=0.02):
    """
    Replicate parameters for each attention head.

    From official TTT implementation.
    """
    flat_params = flax.traverse_util.flatten_dict(params, sep="/")
    for k in flat_params.keys():
        new_shape = (self.num_heads, *flat_params[k].shape)
        if "scale" in k:
            p = self.param(k, jax.nn.initializers.ones, new_shape, param_dtype)
        elif "kernel" in k:
            if kernel_init == "normal":
                initializer = nn.initializers.normal(std)
            elif kernel_init == "zeros":
                initializer = nn.initializers.zeros
            elif kernel_init == "ones":
                initializer = nn.initializers.ones
            elif kernel_init == "layer_norm":
                # For layer norm, use ones initializer
                initializer = nn.initializers.ones
            else:
                raise NotImplementedError(f"Initializer {kernel_init} Not Implemented.")
            p = self.param(k, initializer, new_shape, param_dtype)
        else:
            p = self.param(k, jax.nn.initializers.zeros, new_shape, param_dtype)
        flat_params[k] = p
    params_init = flax.traverse_util.unflatten_dict(flat_params, sep="/")
    return params_init


class LayerNormTemplate(nn.Module):
    """LayerNorm template for multi-head parameters."""

    name: str
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(name=self.name, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x


class LinearLayerTemplate(nn.Module):
    """Linear layer template for multi-head parameters."""

    width: int
    use_bias: bool
    name: str
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.width, use_bias=self.use_bias, name=self.name,
            dtype=self.dtype, param_dtype=self.param_dtype
        )(x)
        return x


class TTTLayer(nn.Module):
    """
    Test-Time Training Layer.

    Based on official TTT-LM-JAX implementation (TTTLinear variant).

    Key features:
    - Self-supervised learning with target = XV - XK
    - Mini-batch processing (16 tokens)
    - Causal masking with jnp.tril
    - Learnable learning rate (per-head, position-dependent)
    - RoPE (Rotary Position Embedding)
    - Causal convolution for Q/K
    - GLU-style gating

    Attributes:
        config: TTT configuration
    """

    config: TTTConfig

    def setup(self):
        """Initialize TTT layer components."""
        self.width = self.config.hidden_dim
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.mini_batch_size = self.config.mini_batch_size
        # Don't fix seq_shape at setup - compute dynamically based on input length

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            self.mini_batch_size * 2,
            theta=self.config.rope_theta,
            dtype=self.config.dtype
        )

        # Q, K, V, O projections
        self.setup_qkvo()

        # Token position indices for learning rate
        self.setup_token_idx()

        # Learnable learning rate
        self.setup_ttt_lr_gate()

        # Per-head TTT normalization
        self.ttt_norm = LayerNormTemplate(dtype=self.config.dtype, param_dtype=self.config.dtype, name="ttt_norm")
        ttt_norm_params = self.ttt_norm.init(jax.random.PRNGKey(0), jnp.ones([1, self.head_dim]))["params"]
        self.ttt_norm_params = get_multi_head_params(
            self, ttt_norm_params, param_dtype=self.config.dtype, kernel_init="layer_norm"
        )

        # Post-normalization
        self.post_norm = nn.LayerNorm(dtype=self.config.dtype, param_dtype=self.config.dtype)

        # Fast weights for TTT (per-head)
        self.W1 = self.param(
            "ttt_dense_0",
            nn.initializers.normal(self.config.initializer_range),
            (self.num_heads, self.head_dim, self.head_dim),
            self.config.dtype,
        )
        self.b1 = self.param(
            "ttt_bias_0",
            nn.initializers.zeros,
            (self.num_heads, 1, self.head_dim),
            self.config.dtype
        )
        self.ttt_params = (self.W1, self.b1)

    def setup_qkvo(self):
        """Setup Q, K, V, O projections with causal convolution."""
        # Q projection with causal convolution
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # Causal convolutions for Q and K
        self.conv_q = nn.Conv(
            self.config.hidden_dim,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_dim,
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,
        )
        self.conv_k = nn.Conv(
            self.config.hidden_dim,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_dim,
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,
        )

        # V projection
        self.wv = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # Output projection
        self.wo = nn.Dense(
            self.width,
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # Gating projection (GLU-style)
        self.wg = nn.Dense(
            self.width,
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def setup_token_idx(self):
        """Setup token position indices for learning rate."""
        self.token_idx = 1.0 / jnp.arange(1, self.mini_batch_size + 1, dtype=jnp.float32)
        self.learnable_token_idx = self.param(
            "learnable_token_idx",
            nn.initializers.zeros,
            (self.mini_batch_size,),
            jnp.float32
        )

    def setup_ttt_lr_gate(self):
        """Setup learnable learning rate (per-head)."""
        self.learnable_ttt_lr = LinearLayerTemplate(
            width=1,
            use_bias=True,
            name="learnable_ttt_lr",
            dtype=self.config.dtype,
            param_dtype=self.config.dtype
        )
        learnable_ttt_lr_params = self.learnable_ttt_lr.init(
            jax.random.PRNGKey(0), jnp.ones([1, self.width])
        )["params"]
        self.learnable_ttt_lr_params = get_multi_head_params(
            self,
            learnable_ttt_lr_params,
            param_dtype=self.config.dtype,
            kernel_init="normal",
            std=self.config.initializer_range,
        )

    def _split_heads(self, hidden_states):
        """Reshape to multi-head format."""
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _split_mini_batches(self, hidden_states):
        """Split sequence into mini-batches."""
        B, N, num_head, head_dim = hidden_states.shape
        n_mini_batch = N // self.mini_batch_size
        # Dynamically compute seq_shape based on actual sequence length
        seq_shape = (n_mini_batch, self.mini_batch_size)
        hidden_states = hidden_states.reshape(B, *seq_shape, self.num_heads, self.head_dim).transpose(
            0, 3, 1, 2, 4
        )
        return hidden_states

    def get_qkv_projections(self, batch):
        """Get Q, K, V projections with causal convolution."""
        xqk = self.wq(batch)
        XV = self.wv(batch)
        XQ = self.conv_q(xqk)
        XK = self.conv_k(xqk)
        return XQ, XK, XV

    def get_eta(self, X):
        """
        Compute learnable learning rate (per-head, position-dependent).

        From official TTT implementation.
        """
        # Per-head learnable learning rate
        learnable_ttt_lr = vmap(
            lambda x, p: self.learnable_ttt_lr.apply({"params": p}, x),
            axis_name="head",
            in_axes=[None, 0],
            out_axes=1
        )(X, self.learnable_ttt_lr_params)
        learnable_ttt_lr = nn.sigmoid(learnable_ttt_lr)
        learnable_ttt_lr = learnable_ttt_lr.transpose(0, 1, 2, 4, 3)

        # Position-dependent base learning rate
        token_idx = self.learnable_token_idx + self.token_idx
        token_idx = jnp.clip(token_idx, a_min=0.0)

        # Combined learning rate
        eta = (
            (self.config.ttt_base_lr * token_idx).reshape(1, 1, 1, token_idx.shape[0], -1)
            * learnable_ttt_lr
            / self.head_dim
        )
        return eta

    def get_ttt_inputs(self, batch, position_ids):
        """Prepare inputs for TTT processing."""
        B, N, F = batch.shape
        n_mini_batch = N // self.mini_batch_size
        # Dynamically compute seq_shape based on actual input length
        seq_shape = (n_mini_batch, self.mini_batch_size)
        X = batch.reshape(B, *seq_shape, self.width)

        # Get Q, K, V projections
        XQ, XK, XV = self.get_qkv_projections(batch)

        # Compute TTT statistics if requested
        if self.config.output_ttt_stats:
            XV_last_in_mini_batch = XV[:, :: self.mini_batch_size, ...].reshape(
                B, n_mini_batch, self.num_heads, self.head_dim
            )
            XK_last_in_mini_batch = XK[:, :: self.mini_batch_size, ...].reshape(
                B, n_mini_batch, self.num_heads, self.head_dim
            )
            ssl_tgt_last_in_mini_batch = XV_last_in_mini_batch - XK_last_in_mini_batch
            ssl_tgt_mean = (XV - XK).mean(axis=1, keepdims=True).reshape(B, 1, self.num_heads, self.head_dim)
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

    def apply_gate(self, hidden_states, ttt_output):
        """Apply GLU-style gating."""
        y = self.wg(hidden_states)
        y = nn.gelu(y)
        output = y * ttt_output
        return output

    def project_ttt_outputs(self, XQW_batch):
        """Project TTT outputs."""
        z_batch = self.wo(XQW_batch)
        return z_batch

    def process_mini_batch(
        self,
        XQ_mini_batch,
        XK_mini_batch,
        XV_mini_batch,
        eta_mini_batch,
        ttt_params_init,
        ttt_params_mini_batch_init,
        ttt_norm_params,
    ):
        """
        Process a single mini-batch with TTT.

        This is the core TTT algorithm from the official implementation.

        Key steps:
        1. Forward pass with current weights
        2. Compute self-supervised loss: ||f(XK) - (XV - XK)||²
        3. Compute gradients via VJP
        4. Update weights with causal masking
        5. Generate output with updated weights
        """
        W1_init, b1_init = ttt_params_mini_batch_init
        square_eta_mini_batch = eta_mini_batch[: self.mini_batch_size]
        last_eta_in_mini_batch = eta_mini_batch[-1][:, None]

        X1 = XK_mini_batch

        # Forward pass
        Z1 = X1 @ W1_init + b1_init
        ttt_norm_out, ttt_norm_vjp = jax.vjp(
            lambda z: self.ttt_norm.apply({"params": ttt_norm_params}, z), Z1
        )

        # Self-supervised target: XV - XK (CRITICAL!)
        ssl_target = XV_mini_batch - XK_mini_batch

        # Gradient computation
        grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target
        grad_l_wrt_Z1 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0]

        # Calculate TTT loss using W_init of the current mini-batch
        if self.config.output_ttt_stats:
            ttt_loss_mse_step_0 = (grad_l_wrt_ttt_norm_out[-1] ** 2).mean()
        else:
            ttt_loss_mse_step_0 = None

        # Calculate TTT loss using W_init of the entire sequence
        if self.config.output_ttt_stats:
            W1_0, b1_0 = ttt_params_init
            Z1_0 = X1 @ W1_0 + b1_0
            ttt_norm_out_0 = self.ttt_norm.apply({"params": ttt_norm_params}, Z1_0)
            ttt_loss_mse_init = ((ttt_norm_out_0 - ssl_target)[-1] ** 2).mean()
        else:
            ttt_loss_mse_init = None

        # Causal processing with updated weights
        X1_bar = XQ_mini_batch
        Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0))  # Causal mask!
        b1_bar = b1_init - (square_eta_mini_batch * jnp.tril(jnp.ones_like(Attn1))) @ grad_l_wrt_Z1
        Z1_bar = X1_bar @ W1_init - (square_eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
        ttt_norm_out_bar = self.ttt_norm.apply({"params": ttt_norm_params}, Z1_bar)

        # Output with residual connection
        output_mini_batch = X1_bar + ttt_norm_out_bar

        # Weight update for next mini-batch
        W1_bar_last = W1_init - (last_eta_in_mini_batch * X1).transpose(1, 0) @ grad_l_wrt_Z1
        b1_bar_last = b1_init - jnp.sum(last_eta_in_mini_batch * grad_l_wrt_Z1, axis=0, keepdims=True)

        # Calculate ttt loss using the updated W_init by the current mini-batch
        if self.config.output_ttt_stats:
            X1_last_fwd_new = X1[-1:] @ W1_bar_last + b1_bar_last
            X1_last_fwd_new = self.ttt_norm.apply({"params": ttt_norm_params}, X1_last_fwd_new)
            ttt_loss_mse_step_1 = ((X1_last_fwd_new - ssl_target[-1:]) ** 2).mean()
        else:
            ttt_loss_mse_step_1 = None

        ttt_params_mini_batch_new = (W1_bar_last, b1_bar_last)

        return (
            ttt_params_mini_batch_new,
            (output_mini_batch, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1),
        )

    def ttt(self, XQ, XK, XV, eta, input_ids):
        """
        Perform Test-Time Training across all mini-batches.

        From official TTT implementation.
        """
        B, N = XV.shape[0], XV.shape[2] * XV.shape[3]

        @partial(vmap, axis_name="batch")
        def update_embed(XQ, XK, XV, eta):
            @partial(vmap, axis_name="head")
            def parallelize_over_heads(XQ, XK, XV, eta, ttt_params_init, ttt_norm_params):
                def compute_mini_batch(ttt_params_mini_batch_init, inputs):
                    XQ_mini_batch = inputs["XQ"]
                    XK_mini_batch = inputs["XK"]
                    XV_mini_batch = inputs["XV"]
                    eta_mini_batch = inputs["eta"]

                    ttt_params_last_in_mini_batch, outputs = self.process_mini_batch(
                        XQ_mini_batch,
                        XK_mini_batch,
                        XV_mini_batch,
                        eta_mini_batch,
                        ttt_params_init,
                        ttt_params_mini_batch_init,
                        ttt_norm_params,
                    )
                    return ttt_params_last_in_mini_batch, outputs

                inputs = {"XQ": XQ, "XK": XK, "XV": XV, "eta": eta}

                # Use scan with remat for memory efficiency
                # Groups mini-batches and applies gradient checkpointing
                _, outputs = scan_remat_every_n_iterations_scan(
                    compute_mini_batch,
                    self.config.remat_mini_batch_group_size,
                    ttt_params_init,
                    inputs
                )
                Z, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1 = outputs
                return (Z.reshape(-1, self.head_dim), ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1)

            outputs = parallelize_over_heads(XQ, XK, XV, eta, self.ttt_params, self.ttt_norm_params)
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
        hidden_states,
        mask: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        deterministic: bool = True,
        enable_internal_updates: bool = True,
    ) -> tuple[jnp.ndarray, dict]:
        """
        Apply TTT layer.

        Args:
            hidden_states: Input [batch, seq_len, hidden_dim]
            mask: Attention mask [batch, seq_len] (not used in current implementation)
            position_ids: Position indices for RoPE [batch, seq_len]
            deterministic: Whether to use dropout
            enable_internal_updates: If True, perform TTT updates (always True now)

        Returns:
            output: Transformed sequence [batch, seq_len, hidden_dim]
            ttt_stats: Dictionary with TTT statistics
        """
        # Generate position IDs if not provided
        if position_ids is None:
            batch_size, seq_len = hidden_states.shape[:2]
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

        # Get TTT inputs
        XQ, XK, XV, eta, precompute_stats = self.get_ttt_inputs(hidden_states, position_ids=position_ids)

        # Perform TTT
        Z, ttt_stats = self.ttt(XQ, XK, XV, eta, None)

        # Post-normalization
        Z = self.post_norm(Z)

        # Apply gating
        Z = self.apply_gate(hidden_states, Z)

        # Output projection
        ttt_output = self.project_ttt_outputs(Z)

        # Combine stats
        all_stats = {
            "ttt_loss_init": ttt_stats[0],
            "ttt_loss_step_0": ttt_stats[1],
            "ttt_loss_step_1": ttt_stats[2],
            "ssl_target_variance": precompute_stats[0],
        }

        return ttt_output, all_stats

"""
Test-Time Training Layer implementation in Flax NNX.

Migrated from Linen to NNX while preserving TTT algorithm exactly.
Based on official TTT-LM-JAX implementation.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import vmap
from jax.sharding import PartitionSpec as P
from jax.tree_util import tree_map


def maybe_with_partitioning(
    fn: Callable,
    axis_rules: Optional[Callable] = None,
    axis_rules_args: Tuple[Optional[str], ...] = (),
) -> Callable:
    """Apply sharding partitioning to initializer if axis_rules provided.

    Args:
        fn: Initializer function
        axis_rules: Callable that maps logical axes to PartitionSpec
        axis_rules_args: Arguments to pass to axis_rules (None = replicate on that axis)

    Returns:
        Partitioned initializer or original if no axis_rules
    """
    if axis_rules is None:
        return fn
    return nnx.with_partitioning(fn, axis_rules(*axis_rules_args))


def scan_remat_every_n_iterations_scan(f, n, carry, x):
    """
    Remat every n mini batches for memory efficiency.

    From official TTT implementation.
    Groups mini-batches into chunks of size n, applies jax.checkpoint to reduce
    memory usage during backpropagation.
    """
    x_grouped = tree_map(lambda x: x.reshape((-1, n, *x.shape[1:])), x)

    def inner_scan(c, xs):
        return jax.lax.scan(f, c, xs)

    carry, y_grouped = jax.lax.scan(
        jax.remat(inner_scan, prevent_cse=False), # type: ignore[arg-type]
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
    causal_k: int = 0  # Causal mask diagonal: 0 includes diagonal, -1 excludes it
    # Eta decay rate for exponential position weighting (0 = linear, >0 = exponential)
    # Higher values give more weight to recent positions, fixing gradient misalignment
    # for larger models (350M+). Recommended: 0.3-0.5 for 350M.
    eta_decay_rate: float = 0.0
    # Sharding configuration (for multi-host TPU)
    axis_rules: Optional[Callable[..., P]] = None

    @classmethod
    def for_gpt2(cls, model_size: str = "125m") -> "TTTConfig":
        """TTT config for GPT-2 models."""
        configs: dict[str, dict] = {
            "125m": {"hidden_dim": 768, "num_heads": 12, "head_dim": 64, "eta_decay_rate": 0.0},
            # 350M+ needs eta_decay_rate to fix gradient misalignment across positions
            "350m": {"hidden_dim": 1024, "num_heads": 16, "head_dim": 64, "eta_decay_rate": 0.3},
            "1b": {"hidden_dim": 1280, "num_heads": 20, "head_dim": 64, "eta_decay_rate": 0.3},
            "xl": {"hidden_dim": 1600, "num_heads": 25, "head_dim": 64, "eta_decay_rate": 0.3},
        }
        config = configs.get(model_size, configs["125m"])
        return cls(
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            head_dim=config["head_dim"],
            eta_decay_rate=config.get("eta_decay_rate", 0.0),
        )

    @classmethod
    def for_gemma3_4b(cls, dtype: jnp.dtype = jnp.bfloat16) -> "TTTConfig":
        """TTT config for Gemma 3 4B.

        Gemma 3 4B specs:
        - embed_dim: 2560
        - num_heads: 8
        - head_dim: 256
        """
        return cls(
            hidden_dim=2560,
            num_heads=8,
            head_dim=256,
            dtype=dtype,
            rope_theta=10000.0,  # Use local attention frequency
            mini_batch_size=16,
            max_seq_length=131072,  # Gemma 3 supports 128K context
        )

    @classmethod
    def for_gemma3_12b(cls, dtype: jnp.dtype = jnp.bfloat16) -> "TTTConfig":
        """TTT config for Gemma 3 12B.

        Gemma 3 12B specs:
        - embed_dim: 3840
        - num_heads: 16
        - head_dim: 256
        """
        return cls(
            hidden_dim=3840,
            num_heads=16,
            head_dim=256,
            dtype=dtype,
            rope_theta=10000.0,
            mini_batch_size=16,
            max_seq_length=131072,
        )

    @classmethod
    def for_gemma3_1b(cls, dtype: jnp.dtype = jnp.bfloat16) -> "TTTConfig":
        """TTT config for Gemma 3 1B (for testing)."""
        return cls(
            hidden_dim=1152,
            num_heads=4,
            head_dim=256,
            dtype=dtype,
            rope_theta=10000.0,
            mini_batch_size=16,
            max_seq_length=32768,
        )


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
        return normalized * self.scale[...][head_idx] + self.bias[...][head_idx]


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
        axis_rules = config.axis_rules

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            config.max_seq_length,
            theta=config.rope_theta,
            dtype=config.dtype,
        )

        # Q, K, V, O projections with sharding
        # Sharding: [hidden_dim, num_heads * head_dim] -> ('embed', 'kv')
        self.wq = nnx.Linear(
            config.hidden_dim,
            config.num_heads * config.head_dim,
            use_bias=False,
            kernel_init=maybe_with_partitioning(
                nnx.initializers.lecun_normal(),
                axis_rules,
                ('embed', 'kv'),
            ),
            rngs=rngs,
        )
        self.wv = nnx.Linear(
            config.hidden_dim,
            config.num_heads * config.head_dim,
            use_bias=False,
            kernel_init=maybe_with_partitioning(
                nnx.initializers.lecun_normal(),
                axis_rules,
                ('embed', 'kv'),
            ),
            rngs=rngs,
        )
        # Zero-init output projection to ensure identity at start of training
        # Sharding: [num_heads * head_dim, hidden_dim] -> ('kv', 'embed')
        self.wo = nnx.Linear(
            config.num_heads * config.head_dim,
            config.hidden_dim,
            use_bias=False,
            kernel_init=maybe_with_partitioning(
                nnx.initializers.zeros,
                axis_rules,
                ('kv', 'embed'),
            ),
            rngs=rngs,
        )
        # Gating projection: [hidden_dim, num_heads * head_dim] -> ('embed', 'kv')
        # Note: Output dimension must match TTT output (num_heads * head_dim)
        self.wg = nnx.Linear(
            config.hidden_dim,
            config.num_heads * config.head_dim,
            use_bias=False,
            kernel_init=maybe_with_partitioning(
                nnx.initializers.lecun_normal(),
                axis_rules,
                ('embed', 'kv'),
            ),
            rngs=rngs,
        )

        # Causal convolutions for Q and K
        # Note: NNX Conv doesn't directly support CAUSAL padding like Linen
        # We'll handle causal masking manually in forward pass
        # Conv is applied after Q/K projection, so use num_heads * head_dim
        qk_dim = config.num_heads * config.head_dim
        self.conv_q = nnx.Conv(
            in_features=qk_dim,
            out_features=qk_dim,
            kernel_size=(config.conv_width,),
            feature_group_count=qk_dim,  # Depthwise
            rngs=rngs,
        )
        self.conv_k = nnx.Conv(
            in_features=qk_dim,
            out_features=qk_dim,
            kernel_size=(config.conv_width,),
            feature_group_count=qk_dim,  # Depthwise
            rngs=rngs,
        )

        # Token position indices for learning rate
        self.token_idx = 1.0 / jnp.arange(1, self.mini_batch_size + 1, dtype=jnp.float32)
        self.learnable_token_idx = nnx.Param(jnp.zeros((self.mini_batch_size,), dtype=jnp.float32))

        # Exponential decay weight for position-dependent eta
        # This gives more weight to recent positions, fixing gradient misalignment for larger models
        # position 15: exp(0) = 1.0, position 0: exp(-15 * decay_rate)
        position_offset = jnp.arange(self.mini_batch_size) - (self.mini_batch_size - 1)
        self.exp_decay_weight = jnp.exp(config.eta_decay_rate * position_offset.astype(jnp.float32))

        # Learnable learning rate (per-head)
        # Use Dict with string keys to avoid int/str comparison issues in sorting
        self.learnable_ttt_lr_layers = nnx.Dict({
            str(i): nnx.Linear(config.hidden_dim, 1, rngs=rngs) 
            for i in range(config.num_heads)
        })

        # Per-head TTT normalization
        self.ttt_norm = TTTLayerNorm(config.num_heads, config.head_dim, 1e-5, rngs)

        # Post-normalization
        self.post_norm = nnx.LayerNorm(config.num_heads * config.head_dim, epsilon=1e-5, rngs=rngs)

        # Fast weights for TTT (per-head)
        # Sharding: [num_heads, head_dim, head_dim] -> ('kv', None, None) - shard across heads
        normal_init = maybe_with_partitioning(
            nnx.initializers.normal(config.initializer_range),
            axis_rules,
            ('kv', None, None),
        )
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
        B, N, C = batch.shape
        
        # Flatten for Linear layers
        batch_flat = batch.astype(jnp.float32).reshape(-1, C)
        
        xqk = self.wq(batch_flat)
        xqk = xqk.reshape(B, N, -1)  # Reshape back
        
        XV = self.wv(batch_flat)
        XV = XV.reshape(B, N, -1)  # Reshape back
        
        XQ = self._apply_causal_conv(self.conv_q, xqk)
        XK = self._apply_causal_conv(self.conv_k, xqk)
        return XQ, XK, XV

    def get_eta(self, X: jax.Array, gating_scale: Optional[jax.Array] = None) -> jax.Array:
        """Compute learnable learning rate (per-head, position-dependent)."""
        # Per-head learnable learning rate
        # X shape: [B, n_mini_batch, mini_batch_size, width]
        B, N_mini, Mini, W = X.shape
        X_flat = X.astype(jnp.float32).reshape(-1, W)
        
        learnable_lrs = []
        for head_idx in range(len(self.learnable_ttt_lr_layers)):
            lr_layer = self.learnable_ttt_lr_layers[str(head_idx)]
            lr = lr_layer(X_flat)  # [B*n*m, 1]
            lr = lr.reshape(B, N_mini, Mini, 1)
            learnable_lrs.append(lr)

        learnable_ttt_lr = jnp.stack(learnable_lrs, axis=1)  # [B, num_heads, n_mini_batch, mini_batch_size, 1]
        learnable_ttt_lr = jax.nn.sigmoid(learnable_ttt_lr)

        # Position-dependent base learning rate
        token_idx = self.learnable_token_idx[...] + self.token_idx
        token_idx = jnp.clip(token_idx, a_min=0.0)

        # Apply exponential decay weighting if configured
        # This gives more weight to recent positions, fixing gradient misalignment for larger models
        # The exp_decay_weight is precomputed in __init__ for correct shape
        token_idx = token_idx * self.exp_decay_weight

        # Combined learning rate
        # Scale by head_dim for gradient magnitude normalization (standard TTT approach)
        # Additional 768/hidden_dim scaling prevents overshooting for larger models
        # (350M with hidden_dim=1024 needs 25% smaller eta than 125M with hidden_dim=768)
        hidden_dim_scale = 768.0 / self.config.hidden_dim
        eta = (
            (self.config.ttt_base_lr * token_idx).reshape(1, 1, 1, token_idx.shape[0], -1)
            * learnable_ttt_lr
            * hidden_dim_scale
            / self.head_dim
        )

        # Apply gating scale if provided
        if gating_scale is not None:
            if gating_scale.ndim == 2:
                # [B, 1]
                gating_scale = gating_scale[:, None, None, None, :]
            elif gating_scale.ndim == 3:
                # [B, N_mini, 1]
                gating_scale = gating_scale[:, None, :, None, :]
                
            eta = eta * gating_scale
            
        return eta

    def get_ttt_inputs(
        self, 
        batch: jax.Array, 
        position_ids: jax.Array,
        gating_scale: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Tuple]:
        """Prepare inputs for TTT processing."""
        B, N, F = batch.shape
        if N % self.mini_batch_size != 0:
            raise ValueError(
                f"Sequence length {N} must be divisible by mini_batch_size {self.mini_batch_size} "
                "for TTT mini-batch processing."
            )
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

        # Apply RoPE with true positions (no mini-batch wrapping)
        # Use mini-batch relative positions for RoPE (wrap positions per mini-batch)
        rel_position_ids = position_ids % self.mini_batch_size
        # Check removed for JIT compatibility (modulo guarantees bounds if mini_batch_size < max_seq_len)
        freqs_cis = jnp.take(self.freqs_cis, rel_position_ids, axis=0)
        XQ, XK = apply_rotary_emb(XQ, XK, freqs_cis=freqs_cis, dtype=self.config.dtype)

        # Split into mini-batches
        XQ = self._split_mini_batches(XQ)
        XK = self._split_mini_batches(XK)
        XV = self._split_mini_batches(XV)

        # Get learning rate
        eta = self.get_eta(X, gating_scale=gating_scale)

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
        # last_eta_in_mini_batch was incorrect and is no longer used

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
        Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0), k=self.config.causal_k)  # Causal mask!
        
        # Fix: Scale columns by eta (eta_k), not rows (eta_t)
        # square_eta_mini_batch is [M, 1], we need [1, M] for broadcasting over columns
        eta_row = square_eta_mini_batch.reshape(1, -1)
        
        b1_bar = b1_init - (jnp.tril(jnp.ones_like(Attn1), k=self.config.causal_k) * eta_row) @ grad_l_wrt_Z1
        Z1_bar = X1_bar @ W1_init - (Attn1 * eta_row) @ grad_l_wrt_Z1 + b1_bar
        ttt_norm_out_bar = ttt_norm_fn(Z1_bar)

        # Output with residual connection
        output_mini_batch = X1_bar + ttt_norm_out_bar

        # Weight update for next mini-batch
        # Fix: Use all etas in mini-batch, not just the last one
        W1_bar_last = W1_init - (eta_mini_batch * X1).transpose(1, 0) @ grad_l_wrt_Z1
        b1_bar_last = b1_init - jnp.sum(eta_mini_batch * grad_l_wrt_Z1, axis=0, keepdims=True)

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
                XQ, XK, XV, eta, self.W1[...], self.b1[...], head_indices
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
        *,
        train: bool = False,
        gating_scale: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, dict]:
        """Apply TTT layer.

        Args:
            hidden_states: Input [batch, seq_len, hidden_dim]
            mask: Attention mask (not used)
            position_ids: Position indices for RoPE
            gating_scale: Optional scaling factor for learning rate (for differentiable TTT)

        Returns:
            (output, ttt_stats) tuple

        Note:
            TTT layer doesn't use dropout, so no train/eval mode distinction needed.
        """
        del mask, train  # Interface parity with LoRA fast weights
        # Generate position IDs if not provided
        if position_ids is None:
            batch_size, seq_len = hidden_states.shape[:2]
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

        # Get TTT inputs
        XQ, XK, XV, eta, precompute_stats = self.get_ttt_inputs(
            hidden_states, 
            position_ids=position_ids,
            gating_scale=gating_scale,
        )

        # Perform TTT
        Z, ttt_stats = self.ttt(XQ, XK, XV, eta)

        # Post-normalization
        Z = self.post_norm(Z)

        # Apply gating
        B, T, C = hidden_states.shape
        hidden_states_flat = hidden_states.astype(jnp.float32).reshape(-1, C)

        y = self.wg(hidden_states_flat)
        # Gating output is [B*T, num_heads * head_dim]
        y = y.reshape(B, T, self.num_heads * self.head_dim)
        y = jax.nn.gelu(y, approximate=True)
        Z = y * Z

        # Output projection
        Z_flat = Z.astype(jnp.float32).reshape(-1, Z.shape[-1])
        ttt_output = self.wo(Z_flat)
        ttt_output = ttt_output.reshape(B, T, -1)

        # Combine stats
        all_stats = {
            "ttt_loss_init": ttt_stats[0],
            "ttt_loss_step_0": ttt_stats[1],
            "ttt_loss_step_1": ttt_stats[2],
            "ssl_target_variance": precompute_stats[0],
        }

        return ttt_output, all_stats

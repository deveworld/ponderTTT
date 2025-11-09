from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._pytree import tree_map


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def scan(f, init, xs, out):
    carry = init
    num_items = len(xs[next(iter(xs))]) if isinstance(xs, dict) else len(xs[0])
    for i in range(num_items):
        if isinstance(xs, dict):
            x = {key: tensor[i] for key, tensor in xs.items()}
        else:
            x = [elem[i] for elem in xs]
        carry, y = f(carry, x)
        out[i] = y
    return carry, out


def ln_fwd(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    return y


def ln_fused_l2_bwd(
    x: torch.Tensor,
    l2_target: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    d = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / d)
        * (
            d * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )
    return z


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 16,
        base: float = 10000.0,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


@dataclass
class TTTLayerConfig:
    hidden_size: int
    num_attention_heads: int
    mini_batch_size: int
    ttt_base_lr: float
    conv_kernel: int = 4
    share_qk: bool = True
    rope_theta: float = 10000.0
    use_gate: bool = True


class TTTLinear(nn.Module):
    def __init__(self, config: TTTLayerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = self.hidden_size // self.num_heads
        self.mini_batch_size = config.mini_batch_size

        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros(self.mini_batch_size))

        self.share_qk = config.share_qk
        self.conv_kernel = config.conv_kernel
        self._init_qkv()
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.mini_batch_size, base=config.rope_theta)
        self._init_eta_gate()
        self._init_layer_norm()
        self.use_gate = config.use_gate
        if self.use_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.post_norm = nn.LayerNorm(self.hidden_size)

        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def _init_qkv(self):
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        if not self.share_qk:
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        if self.share_qk:
            self.conv_q = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                kernel_size=self.conv_kernel,
                groups=self.hidden_size,
                padding=self.conv_kernel - 1,
            )
            self.conv_k = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                kernel_size=self.conv_kernel,
                groups=self.hidden_size,
                padding=self.conv_kernel - 1,
            )

    def _init_eta_gate(self):
        # Shape: [num_heads, 1, hidden_size] for einsum compatibility
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.normal(0, 0.02, size=(self.num_heads, 1, self.hidden_size))
        )
        self.learnable_ttt_lr_bias = nn.Parameter(torch.zeros(self.num_heads, 1, 1))

    def _init_layer_norm(self):
        ln_weight = nn.LayerNorm(self.head_dim).weight.data
        ln_bias = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight.unsqueeze(0), (self.num_heads, 1)))
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias.unsqueeze(0), (self.num_heads, 1)))

    def _apply_depthwise_conv(self, tensor: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        tensor = tensor.transpose(1, 2)
        out = conv(tensor)
        out = out[:, :, :seq_len]
        out = out.transpose(1, 2)
        return out

    def get_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.share_qk:
            xq = self.q_proj(hidden_states)
            xv = self.v_proj(hidden_states)
            xq_conv = self._apply_depthwise_conv(xq, self.conv_q)
            xk_conv = self._apply_depthwise_conv(xq, self.conv_k)
            XQ = xq_conv
            XK = xk_conv
        else:
            XQ = self.q_proj(hidden_states)
            XK = self.k_proj(hidden_states)
            xv = self.v_proj(hidden_states)
        XV = xv
        return XQ, XK, XV

    def _get_eta(
        self,
        X: torch.Tensor,
        mini_batch_size: int,
        token_scaling: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # X shape: [B, num_mini_batch, mini_batch_size, hidden_size]
        ttt_lr = torch.einsum(
            "bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight
        ) + self.learnable_ttt_lr_bias.reshape(1, -1, 1, 1, 1)
        ttt_lr = torch.sigmoid(ttt_lr)
        # Permute: [B, num_heads, num_mini_batch, mini_batch_size, 1] -> [B, num_heads, num_mini_batch, 1, mini_batch_size]
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.config.ttt_base_lr * ttt_lr / self.head_dim

        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[:mini_batch_size]
        token_idx = torch.clamp_min(token_idx, 0.0)
        base_token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )
        if token_scaling is not None:
            scale = token_scaling.to(X.dtype).reshape(X.shape[0], 1, X.shape[1], mini_batch_size, 1)
            token_eta = base_token_eta * scale
        else:
            token_eta = base_token_eta
        return token_eta, ttt_lr_eta

    def _chunk_for_ttt(
        self,
        hidden_states: torch.Tensor,
        XQ: torch.Tensor,
        XK: torch.Tensor,
        XV: torch.Tensor,
        mini_batch_size: int,
        token_scaling: Optional[torch.Tensor] = None,
    ):
        B, L = hidden_states.shape[:2]
        num_mini_batch = L // mini_batch_size
        X = hidden_states.reshape(B, num_mini_batch, mini_batch_size, self.hidden_size)
        scaling_chunk = None
        if token_scaling is not None:
            scaling_chunk = token_scaling.reshape(B, num_mini_batch, mini_batch_size)

        def reshape_proj(proj: torch.Tensor):
            proj = proj.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            proj = proj.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
            return proj

        XQ = reshape_proj(XQ)
        XK = reshape_proj(XK)
        XV = reshape_proj(XV)
        token_eta, ttt_lr_eta = self._get_eta(X, mini_batch_size, scaling_chunk)
        eta = token_eta * ttt_lr_eta
        return {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }

    def _ttt_inner(
        self,
        chunk_inputs: Dict[str, torch.Tensor],
        mini_batch_size: int,
        init_states: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Inner TTT computation using scan over mini-batches.

        Args:
            chunk_inputs: Dictionary with chunked inputs
            mini_batch_size: Size of each mini-batch
            init_states: Optional initial W1/b1 states to carry forward from previous call
                        If None, initializes from learned parameters

        Returns:
            output: Processed output
            final_states: Final W1 and b1 states (for state carry)
        """
        B = chunk_inputs["XV"].shape[0]
        num_mini_batch = chunk_inputs["XV"].shape[2]
        device = chunk_inputs["XV"].device
        dtype = chunk_inputs["XV"].dtype

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict["W1_states"]
            b1_init = params_dict["b1_states"]
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            eta_mini_batch = inputs["eta"]

            X1 = XK_mini_batch
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch
            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

            attn = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
            tril_eta = torch.tril(eta_mini_batch)
            b1_bar = b1_init - tril_eta @ grad_l_wrt_Z1
            Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * attn) @ grad_l_wrt_Z1 + b1_bar

            last_eta = eta_mini_batch[:, :, -1, :, None]
            W1_last = W1_init - (last_eta * X1).transpose(-1, -2) @ grad_l_wrt_Z1
            b1_last = b1_init - torch.sum(last_eta * grad_l_wrt_Z1, dim=-2, keepdim=True)

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)
            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
            }
            return last_param_dict, XQW_mini_batch

        # Initialize states: either from provided init_states or from learned parameters
        if init_states is not None:
            init_params = init_states
        else:
            init_params = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), (B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), (B, 1, 1, 1)),
            }

        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), chunk_inputs)
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        batch_params_dict, XQW_batch = scan(compute_mini_batch, init_params, inputs, XQW_batch)
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4).reshape(B, num_mini_batch * mini_batch_size, self.hidden_size)
        return XQW_batch, batch_params_dict

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        reminder_len = seq_len % self.mini_batch_size
        num_full = seq_len // self.mini_batch_size

        XQ, XK, XV = self.get_qkv(hidden_states)
        position_ids = torch.arange(seq_len, device=hidden_states.device)[None, :]
        cos, sin = self.rotary_emb(XV.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), position_ids % self.mini_batch_size)
        XQ = XQ.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
        outputs = []
        final_states = None
        if num_full > 0:
            start_len = num_full * self.mini_batch_size
            chunks = self._chunk_for_ttt(
                hidden_states[:, :start_len],
                XQ[:, :, :start_len],
                XK[:, :, :start_len],
                XV[:, :, :start_len],
                self.mini_batch_size,
                None if token_scaling is None else token_scaling[:, :start_len],
            )
            out_chunk, final_states = self._ttt_inner(chunks, self.mini_batch_size)
            outputs.append(out_chunk)
        if reminder_len > 0:
            chunks = self._chunk_for_ttt(
                hidden_states[:, -reminder_len:],
                XQ[:, :, -reminder_len:],
                XK[:, :, -reminder_len:],
                XV[:, :, -reminder_len:],
                reminder_len,
                None if token_scaling is None else token_scaling[:, -reminder_len:],
            )
            # Pass final_states from full mini-batches to remainder
            out_chunk, _ = self._ttt_inner(chunks, reminder_len, init_states=final_states)
            outputs.append(out_chunk)
        output_hidden_states = torch.cat(outputs, dim=1) if outputs else hidden_states.new_zeros(hidden_states.shape)
        output_hidden_states = self.post_norm(output_hidden_states)
        if self.use_gate:
            y = self.g_proj(hidden_states)
            y = F.gelu(y, approximate="tanh")
            output_hidden_states = y * output_hidden_states
        output_hidden_states = self.out_proj(output_hidden_states)
        return output_hidden_states

    def ttt_forward(
        self,
        x: torch.Tensor,
        token_scaling: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(x, token_scaling=token_scaling)
        return out, {}


class TTTLinearSequential(TTTLinear):
    pass


class TTTLinearIndependent(TTTLinear):
    def _ttt_inner(
        self,
        chunk_inputs: Dict[str, torch.Tensor],
        mini_batch_size: int,
        init_states: Optional[Dict[str, torch.Tensor]] = None,
    ):
        B = chunk_inputs["XV"].shape[0]
        num_mini_batch = chunk_inputs["XV"].shape[2]
        device = chunk_inputs["XV"].device
        dtype = chunk_inputs["XV"].dtype

        def compute_mini_batch(params_dict, inputs):
            W1_init = torch.tile(self.W1.unsqueeze(0), (B, 1, 1, 1))
            b1_init = torch.tile(self.b1.unsqueeze(0), (B, 1, 1, 1))

            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            eta_mini_batch = inputs["eta"]

            X1 = XK_mini_batch
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch
            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

            attn = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
            tril_eta = torch.tril(eta_mini_batch)
            b1_bar = b1_init - tril_eta @ grad_l_wrt_Z1
            Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * attn) @ grad_l_wrt_Z1 + b1_bar

            last_eta = eta_mini_batch[:, :, -1, :, None]
            W1_last = W1_init - (last_eta * X1).transpose(-1, -2) @ grad_l_wrt_Z1
            b1_last = b1_init - torch.sum(last_eta * grad_l_wrt_Z1, dim=-2, keepdim=True)

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)
            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
            }
            return last_param_dict, XQW_mini_batch

        init_params = {
            "W1_states": torch.tile(self.W1.unsqueeze(0), (B, 1, 1, 1)),
            "b1_states": torch.tile(self.b1.unsqueeze(0), (B, 1, 1, 1)),
        }

        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), chunk_inputs)
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        batch_params_dict, XQW_batch = scan(compute_mini_batch, init_params, inputs, XQW_batch)
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4).reshape(B, num_mini_batch * mini_batch_size, self.hidden_size)
        return XQW_batch, batch_params_dict


class TTTLinearWithStats(TTTLinear):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = super().forward(x)
        return out, {}

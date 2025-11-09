"""
Transformer with TTT layers for language modeling.

Implements a standard Transformer decoder with option to replace
self-attention layers with TTT layers.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_ttt import HeuristicAdaptiveTTT
from .ttt_linear import TTTLayerConfig, TTTLinear, TTTLinearSequential, TTTLinearIndependent


class TransformerConfig:
    """Configuration for Transformer with TTT."""

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        # TTT-specific
        use_ttt: bool = True,
        ttt_layer_idx: Optional[int] = 3,  # Which layer to replace with TTT (0-indexed)
        ttt_layer_indices: Optional[List[int]] = None,  # Multiple TTT layers
        ttt_dim: int = 256,
        ttt_iterations: int = 2,
        ttt_lr: float = 0.01,
        use_sequential_ttt: bool = True,  # Use sequential TTT by default
        ttt_mini_batch_size: int = 16,
        ttt_conv_kernel: int = 4,
        ttt_base_lr: float = 1.0,
        ttt_share_qk: bool = True,
        ttt_rope_theta: float = 10000.0,
        ttt_use_gate: bool = True,
        # Adaptive TTT
        use_adaptive_ttt: bool = False,
        ttt_difficulty_metric: str = "loss",
        ttt_buckets: Optional[List[int]] = None,
        ttt_target_distribution: Optional[List[float]] = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # TTT config
        self.use_ttt = use_ttt

        # Support both single layer and multiple layers
        if ttt_layer_indices is not None:
            self.ttt_layer_indices = ttt_layer_indices
        elif ttt_layer_idx is not None:
            self.ttt_layer_indices = [ttt_layer_idx]
        else:
            self.ttt_layer_indices = [3]

        self.ttt_dim = ttt_dim
        self.ttt_iterations = ttt_iterations
        self.ttt_lr = ttt_lr
        self.ttt_mini_batch_size = ttt_mini_batch_size
        self.ttt_conv_kernel = ttt_conv_kernel
        self.ttt_base_lr = ttt_base_lr
        self.ttt_share_qk = ttt_share_qk
        self.ttt_rope_theta = ttt_rope_theta
        self.ttt_use_gate = ttt_use_gate

        # Adaptive TTT
        self.use_adaptive_ttt = use_adaptive_ttt
        self.ttt_difficulty_metric = ttt_difficulty_metric
        self.ttt_buckets = ttt_buckets or [1, 2, 4]
        self.ttt_target_distribution = ttt_target_distribution or [0.3, 0.4, 0.3]

        # TTT variant (Sequential is default)
        self.use_sequential_ttt = use_sequential_ttt


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim: int = hidden_dim
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if mask is None:
            # Create causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output: torch.Tensor = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        output = output.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.out_proj(output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer decoder block (self-attention + FFN)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_ttt: bool = False,
        ttt_config: Optional[dict] = None,
        use_adaptive_ttt: bool = False,
    ):
        super().__init__()
        self.use_ttt: bool = use_ttt
        self.use_adaptive_ttt: bool = use_adaptive_ttt

        # Layer 1: Self-attention or TTT
        attention_module: Union[MultiHeadAttention, TTTLinear, TTTLinearSequential, HeuristicAdaptiveTTT]
        base_ttt: Union[TTTLinear, TTTLinearSequential]
        if use_ttt:
            # Use TTT layer instead of self-attention
            if ttt_config is None:
                ttt_config = {}

            # Use Sequential TTT by default
            use_sequential = ttt_config.get("use_sequential_ttt", True)

            layer_cfg = TTTLayerConfig(
                hidden_size=hidden_dim,
                num_attention_heads=num_heads,
                mini_batch_size=ttt_config.get("mini_batch_size", 16),
                ttt_base_lr=ttt_config.get("ttt_base_lr", 1.0),
                conv_kernel=ttt_config.get("conv_kernel", 4),
                share_qk=ttt_config.get("share_qk", True),
                rope_theta=ttt_config.get("rope_theta", 10000.0),
                use_gate=ttt_config.get("use_gate", True),
            )

            if use_sequential:
                base_ttt = TTTLinearSequential(layer_cfg)
            else:
                base_ttt = TTTLinearIndependent(layer_cfg)

            if use_adaptive_ttt:
                attention_module = HeuristicAdaptiveTTT(
                    base_ttt=base_ttt,
                    difficulty_metric=ttt_config.get("difficulty_metric", "entropy"),
                    buckets=ttt_config.get("buckets", [1, 2, 4]),
                    target_distribution=ttt_config.get("target_distribution", [0.3, 0.4, 0.3]),
                    auto_calibrate=True,
                )
            else:
                attention_module = base_ttt
        else:
            # Standard self-attention
            attention_module = MultiHeadAttention(hidden_dim, num_heads, dropout)

        self.attention = attention_module

        self.ln1 = nn.LayerNorm(hidden_dim)

        # Layer 2: Feed-forward
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            logits: Optional logits for adaptive TTT
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, hidden_dim)
            stats: Optional statistics from TTT
        """
        stats = None

        # Self-attention or TTT
        residual = x
        x = self.ln1(x)

        if self.use_ttt:
            if self.use_adaptive_ttt:
                assert isinstance(self.attention, HeuristicAdaptiveTTT)
                attn_out, stats = self.attention.forward_adaptive(x, logits=logits)
            else:
                if isinstance(self.attention, (TTTLinear, TTTLinearSequential)):
                    attn_out, stats = self.attention.ttt_forward(x)
                else:
                    attn_out, stats = self.attention(x), {}
        else:
            assert isinstance(self.attention, MultiHeadAttention)
            attn_out = self.attention(x, mask)

        x = residual + self.dropout(attn_out)

        # Feed-forward
        residual = x
        x = self.ln2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)

        return x, stats


class TransformerTTT(nn.Module):
    """
    Transformer decoder with optional TTT layers for language modeling.

    Args:
        config: TransformerConfig
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config: TransformerConfig = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Position embeddings
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks: nn.ModuleList = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            # Decide if this layer should use TTT
            use_ttt = config.use_ttt and (layer_idx in config.ttt_layer_indices)

            ttt_config = {
                "ttt_dim": config.ttt_dim,
                "ttt_iterations": config.ttt_iterations,
                "ttt_lr": config.ttt_lr,
                "mini_batch_size": config.ttt_mini_batch_size,
                "conv_kernel": config.ttt_conv_kernel,
                "ttt_base_lr": config.ttt_base_lr,
                "share_qk": config.ttt_share_qk,
                "rope_theta": config.ttt_rope_theta,
                "use_gate": config.ttt_use_gate,
                "difficulty_metric": config.ttt_difficulty_metric,
                "buckets": config.ttt_buckets,
                "target_distribution": config.ttt_target_distribution,
                "use_sequential_ttt": config.use_sequential_ttt,
            }

            block = TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                use_ttt=use_ttt,
                ttt_config=ttt_config,
                use_adaptive_ttt=config.use_adaptive_ttt,
            )

            self.blocks.append(block)

        # Output layer norm and LM head
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights (common practice)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len)
            labels: Optional labels for computing loss
            return_stats: Whether to return TTT statistics

        Returns:
            Dictionary with 'logits', optional 'loss', and optional 'stats'
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + pos_emb)

        # Pre-check if any blocks need entropy metric
        # If so, we'll cache logits computation to avoid redundant LM head calls
        needs_entropy = any(
            block.use_ttt and block.use_adaptive_ttt and
            isinstance(block.attention, HeuristicAdaptiveTTT) and
            hasattr(block.attention, 'difficulty_metric') and
            block.attention.difficulty_metric in {'entropy', 'combined'}
            for block in self.blocks
        )

        # Apply transformer blocks
        all_stats = []
        cached_logits_state = None  # Cache for (hidden_state, logits) pair
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            # For adaptive TTT with entropy metric, compute logits from current hidden state
            logits_for_block = None
            if block.use_ttt and block.use_adaptive_ttt:
                # Check if this block uses entropy metric
                if isinstance(block.attention, HeuristicAdaptiveTTT) and \
                   hasattr(block.attention, 'difficulty_metric') and \
                   block.attention.difficulty_metric in {'entropy', 'combined'}:
                    # Check if we can reuse cached logits
                    # (only valid if hidden state hasn't changed)
                    if cached_logits_state is not None:
                        cached_x, cached_logits = cached_logits_state
                        # Simple pointer equality check - x should be same object if unchanged
                        if cached_x is x:
                            logits_for_block = cached_logits

                    # Compute new logits if not cached
                    if logits_for_block is None:
                        with torch.no_grad():
                            logits_for_block = self.lm_head(self.ln_f(x))
                            # Cache for potential reuse by next block
                            # (though typically x changes after each block)
                            cached_logits_state = (x, logits_for_block)

                x, stats = block(x, logits=logits_for_block)
                # Invalidate cache since x has changed
                cached_logits_state = None
            else:
                x, stats = block(x)

            if stats is not None:
                all_stats.append(stats)

        # Final layer norm and LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            # logits: [batch, seq_len, vocab]
            # labels: [batch, seq_len]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        result = {"logits": logits}

        if loss is not None:
            result["loss"] = loss

        if return_stats and all_stats:
            result["ttt_stats"] = all_stats

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Simple greedy/sampling generation.

        Args:
            input_ids: (batch, seq_len) starting tokens
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top-k tokens

        Returns:
            generated: (batch, max_length) generated tokens
        """
        self.eval()
        generated = input_ids

        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(generated)
            logits = outputs["logits"]

            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = (
                    next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

        return generated

"""
Transformer with TTT layers for language modeling.

Implements a standard Transformer decoder with option to replace
self-attention layers with TTT layers.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_ttt import HeuristicAdaptiveTTT
from .ttt_linear import TTTLinear


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
        ttt_layer_idx: int = 3,  # Which layer to replace with TTT (0-indexed)
        ttt_dim: int = 256,
        ttt_iterations: int = 2,
        ttt_lr: float = 0.01,
        # Adaptive TTT
        use_adaptive_ttt: bool = False,
        ttt_difficulty_metric: str = "entropy",
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
        self.ttt_layer_idx = ttt_layer_idx
        self.ttt_dim = ttt_dim
        self.ttt_iterations = ttt_iterations
        self.ttt_lr = ttt_lr

        # Adaptive TTT
        self.use_adaptive_ttt = use_adaptive_ttt
        self.ttt_difficulty_metric = ttt_difficulty_metric
        self.ttt_buckets = ttt_buckets or [1, 2, 4]
        self.ttt_target_distribution = ttt_target_distribution or [0.3, 0.4, 0.3]


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim  # type: ignore[unresolved-attribute]
        self.num_heads = num_heads  # type: ignore[unresolved-attribute]
        self.head_dim = hidden_dim // num_heads  # type: ignore[unresolved-attribute]

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
        output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
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
        self.use_ttt = use_ttt  # type: ignore[unresolved-attribute]
        self.use_adaptive_ttt = use_adaptive_ttt  # type: ignore[unresolved-attribute]

        # Layer 1: Self-attention or TTT
        if use_ttt:
            # Use TTT layer instead of self-attention
            if ttt_config is None:
                ttt_config = {}

            base_ttt = TTTLinear(
                hidden_dim=hidden_dim,
                ttt_dim=ttt_config.get("ttt_dim", 256),
                num_iterations=ttt_config.get("ttt_iterations", 2),
                learning_rate=ttt_config.get("ttt_lr", 0.01),
            )

            if use_adaptive_ttt:
                self.attention = HeuristicAdaptiveTTT(
                    base_ttt=base_ttt,
                    difficulty_metric=ttt_config.get("difficulty_metric", "entropy"),
                    buckets=ttt_config.get("buckets", [1, 2, 4]),
                    target_distribution=ttt_config.get("target_distribution", [0.3, 0.4, 0.3]),
                    auto_calibrate=True,
                )
            else:
                self.attention = base_ttt  # type: ignore[bad-assignment]
        else:
            # Standard self-attention
            self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)  # type: ignore[bad-assignment]

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
                attn_out, stats = self.attention.forward_adaptive(x, logits=logits)  # type: ignore[call-non-callable]
            else:
                attn_out, stats = self.attention.ttt_forward(x)  # type: ignore[call-non-callable]
        else:
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
        self.config = config  # type: ignore[unresolved-attribute]

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Position embeddings
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            # Decide if this layer should use TTT
            use_ttt = config.use_ttt and (layer_idx == config.ttt_layer_idx)

            ttt_config = {
                "ttt_dim": config.ttt_dim,
                "ttt_iterations": config.ttt_iterations,
                "ttt_lr": config.ttt_lr,
                "difficulty_metric": config.ttt_difficulty_metric,
                "buckets": config.ttt_buckets,
                "target_distribution": config.ttt_target_distribution,
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

        # Apply transformer blocks
        all_stats = []
        for block in self.blocks:
            # For adaptive TTT, compute logits if needed
            if block.use_ttt and block.use_adaptive_ttt:  # type: ignore[not-callable]
                # Compute logits from current hidden state for entropy metric
                block_logits = self.lm_head(self.ln_f(x))
                x, stats = block(x, logits=block_logits)
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

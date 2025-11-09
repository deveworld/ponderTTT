"""
Transformer with Iterative TTT layers.

Complete redesign using explicit gradient descent with learned halting policies.
Fixes architecture variance issues from analytic TTT implementation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .iterative_ttt import IterativeTTTLayer
from .halting_policy import HaltingPolicyNetwork, MultiGranularityRouter


@dataclass
class IterativeTransformerConfig:
    """Configuration for Transformer with Iterative TTT."""

    # Model architecture (FIXED - same for all experiments)
    vocab_size: int = 50257  # GPT-2 tokenizer
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1

    # Iterative TTT configuration
    use_iterative_ttt: bool = True
    ttt_layer_indices: List[int] = None  # Will default to [2, 3, 4]
    fast_weight_hidden_dim: int = 64
    ttt_base_lr: float = 0.1
    max_steps: int = 8
    ttt_loss_type: str = "reconstruction"  # or "prediction"
    use_sequential: bool = True  # Carry forward fast-weight states

    # Halting policy configuration
    use_learned_policy: bool = False
    step_options: List[int] = None  # Will default to [1, 2, 4, 8]
    policy_use_lstm: bool = True
    policy_gumbel_tau: float = 1.0
    policy_pooling: str = 'none'  # Batch pooling: 'none' (per-token), 'mean', 'max'

    # Multi-granularity routing
    use_multi_granularity: bool = False

    # Training configuration
    lambda_compute: float = 0.0  # Compute regularization weight
    target_avg_steps: Optional[float] = None  # Target budget for fine-tuning

    # REINFORCE configuration
    gamma: float = 0.99  # Discount factor for temporal credit assignment
    baseline_momentum: float = 0.99  # EMA momentum for baseline update

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.ttt_layer_indices is None:
            self.ttt_layer_indices = [2, 3, 4]
        if self.step_options is None:
            self.step_options = [1, 2, 4, 8]


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention (for non-TTT layers)."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch_size, seq_len, self.hidden_dim)
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


class IterativeTTTBlock(nn.Module):
    """
    Transformer block with iterative TTT layer.

    Replaces self-attention with iterative gradient descent TTT.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        ttt_config: Dict,
        use_halting_policy: bool = False,
    ):
        super().__init__()

        self.use_halting_policy = use_halting_policy
        self.policy_pooling = ttt_config.get('policy_pooling', 'none')

        # Iterative TTT layer
        self.ttt_layer = IterativeTTTLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            fast_weight_hidden_dim=ttt_config.get('fast_weight_hidden_dim', 64),
            base_lr=ttt_config.get('base_lr', 0.1),
            max_steps=ttt_config.get('max_steps', 8),
            ttt_loss_type=ttt_config.get('loss_type', 'reconstruction'),
            use_sequential=ttt_config.get('use_sequential', True),
        )

        # Always create halting policy for parameter fairness
        # Baseline models have policy but don't use it (override with num_steps)
        self.halting_policy = HaltingPolicyNetwork(
            hidden_dim=hidden_dim,
            step_options=ttt_config.get('step_options', [1, 2, 4, 8]),
            use_lstm=ttt_config.get('policy_use_lstm', True),
            gumbel_tau=ttt_config.get('policy_gumbel_tau', 1.0),
        )

        self.ln1 = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            num_steps: Optional fixed step counts (seq_len,)
            return_stats: Whether to return statistics

        Returns:
            output: Output tensor
            stats: Optional statistics (includes log_probs for REINFORCE)
        """
        stats = {}
        log_probs = None

        # TTT layer (replaces self-attention)
        residual = x
        x = self.ln1(x)

        # Determine step counts
        if num_steps is not None:
            # Use provided step counts (for baselines)
            layer_steps = num_steps
            log_probs = None
        elif self.use_halting_policy:
            # Use learned policy with REINFORCE
            layer_steps, log_probs = self.halting_policy(
                x,
                return_probs=True,
                pooling=self.policy_pooling
            )
        else:
            # Use max steps (shouldn't reach here in practice)
            layer_steps = None
            log_probs = None

        # Forward through TTT
        ttt_out, ttt_stats, _ = self.ttt_layer(
            x,
            num_steps=layer_steps,
            return_stats=return_stats
        )

        x = residual + self.dropout(ttt_out)

        # Feed-forward
        residual = x
        x = self.ln2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)

        # Collect statistics
        if return_stats or log_probs is not None:
            if ttt_stats is not None:
                stats.update(ttt_stats)
            if log_probs is not None:
                stats['log_probs'] = log_probs

        return x, stats if stats else None


class StandardTransformerBlock(nn.Module):
    """Standard transformer block (for non-TTT layers)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass."""
        # Self-attention
        residual = x
        x = self.ln1(x)
        attn_out = self.attention(x)
        x = residual + self.dropout(attn_out)

        # Feed-forward
        residual = x
        x = self.ln2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)

        return x, None


class IterativeTransformerTTT(nn.Module):
    """
    Transformer with Iterative TTT layers.

    Key features:
    - Fixed architecture (no mini_batch_size dependence)
    - True per-token iteration control
    - Learned halting policies
    - Optional multi-granularity routing
    - Compute regularization
    - Parameter-matched baselines via DummyPolicy
    """

    def __init__(self, config: IterativeTransformerConfig):
        super().__init__()

        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Build transformer blocks
        self.blocks = nn.ModuleList()

        for layer_idx in range(config.num_layers):
            if config.use_iterative_ttt and (layer_idx in config.ttt_layer_indices):
                # Iterative TTT block
                ttt_config = {
                    'fast_weight_hidden_dim': config.fast_weight_hidden_dim,
                    'base_lr': config.ttt_base_lr,
                    'max_steps': config.max_steps,
                    'loss_type': config.ttt_loss_type,
                    'use_sequential': config.use_sequential,
                    'step_options': config.step_options,
                    'policy_use_lstm': config.policy_use_lstm,
                    'policy_gumbel_tau': config.policy_gumbel_tau,
                    'policy_pooling': config.policy_pooling,
                }

                block = IterativeTTTBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                    ttt_config=ttt_config,
                    use_halting_policy=config.use_learned_policy,
                )
            else:
                # Standard transformer block
                block = StandardTransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                )

            self.blocks.append(block)

        # Multi-granularity router (optional)
        if config.use_multi_granularity:
            self.router = MultiGranularityRouter(
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dim,
                step_options=config.step_options,
            )
        else:
            self.router = None

        # Output layers
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        # REINFORCE baseline (exponential moving average of loss)
        # Only used when use_learned_policy=True
        if config.use_learned_policy:
            self.register_buffer('baseline', torch.tensor(0.0))
            self.register_buffer('baseline_initialized', torch.tensor(False))
            self.baseline_momentum = config.baseline_momentum
            self.gamma = config.gamma  # Discount factor for temporal credit

        # Initialize
        self.apply(self._init_weights)

    def _compute_monte_carlo_returns(
        self,
        per_token_loss: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """
        Compute Monte Carlo returns with temporal credit assignment.

        For sequential tasks where actions at position t affect future rewards,
        we need to compute discounted cumulative returns:

        G_t = sum_{i=t}^{T-1} gamma^{i-t} * reward[i]

        where reward[i] = -loss[i] (lower loss = higher reward)

        Args:
            per_token_loss: Per-token cross-entropy loss (batch, seq_len)
            gamma: Discount factor (default 0.99)

        Returns:
            returns: Monte Carlo returns for each position (batch, seq_len)
        """
        batch_size, seq_len = per_token_loss.shape

        # Convert loss to reward (negative loss)
        rewards = -per_token_loss  # (batch, seq_len)

        # Compute returns using dynamic programming (backward pass)
        # This is more efficient than the naive loop
        returns = torch.zeros_like(rewards)
        returns[:, -1] = rewards[:, -1]  # G_{T-1} = r_{T-1}

        for t in range(seq_len - 2, -1, -1):
            # G_t = r_t + gamma * G_{t+1}
            returns[:, t] = rewards[:, t] + gamma * returns[:, t + 1]

        return returns

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
        num_steps: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Dict:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            labels: Optional labels for computing loss
            num_steps: Optional fixed step counts for all TTT layers (seq_len,)
            return_stats: Whether to return detailed statistics

        Returns:
            Dictionary with logits, loss, and optional stats
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        # Process through transformer blocks
        all_stats = []

        for layer_idx, block in enumerate(self.blocks):
            # Multi-granularity routing (if enabled)
            if self.router is not None:
                use_ttt, layer_steps = self.router(x, layer_idx=layer_idx)
                # If layer disabled, set steps to 0
                if use_ttt.sum() == 0:
                    continue
            else:
                layer_steps = num_steps

            # Forward through block
            x, stats = block(x, num_steps=layer_steps, return_stats=return_stats)

            if stats is not None:
                stats['layer_idx'] = layer_idx
                all_stats.append(stats)

        # LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss
        loss = None
        if labels is not None:
            # Language modeling loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            total_loss = lm_loss

            # REINFORCE policy gradient loss (only when using learned policy)
            if self.config.use_learned_policy and self.training:
                # Collect log_probs from all TTT layers
                log_probs_list = [s.get('log_probs') for s in all_stats if 'log_probs' in s]

                if log_probs_list:
                    # Compute per-token loss for per-token credit assignment
                    per_token_loss = F.cross_entropy(
                        shift_logits.view(-1, self.config.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction='none'
                    ).view(shift_labels.shape)  # (batch, seq_len-1)

                    # Compute Monte Carlo returns with temporal credit assignment
                    # This accounts for the fact that actions at position t affect all future losses
                    # due to sequential fast-weight carry-over
                    returns = self._compute_monte_carlo_returns(
                        per_token_loss.detach(),
                        gamma=self.gamma
                    )  # (batch, seq_len-1)

                    # Update baseline (exponential moving average of mean returns)
                    mean_return = returns.mean()
                    if not self.baseline_initialized:
                        # Initialize baseline with first return value
                        self.baseline.copy_(mean_return.detach())
                        self.baseline_initialized.fill_(True)
                    else:
                        # EMA update: baseline = momentum * baseline + (1 - momentum) * current_return
                        with torch.no_grad():
                            self.baseline.mul_(self.baseline_momentum).add_(
                                mean_return.detach(), alpha=(1 - self.baseline_momentum)
                            )

                    # Compute per-token advantage: advantage = returns - baseline
                    # Higher returns (lower cumulative future loss) = positive advantage
                    per_token_advantage = returns - self.baseline  # (batch, seq_len-1)

                    # Policy gradient loss: -log_prob * advantage
                    policy_loss = 0.0
                    for log_probs in log_probs_list:
                        # log_probs: (batch, seq_len) or (seq_len,) depending on pooling
                        # Align with per_token_advantage shape
                        if log_probs.dim() == 1:
                            # Per-position: expand to batch
                            log_probs_aligned = log_probs.unsqueeze(0).expand(per_token_advantage.shape[0], -1)[:, :-1]
                        else:
                            # Per-token: (batch, seq_len)
                            log_probs_aligned = log_probs[:, :-1]

                        # Per-token credit assignment
                        policy_loss -= (log_probs_aligned * per_token_advantage).mean()

                    # Add policy loss to total
                    total_loss = total_loss + policy_loss

                    if return_stats:
                        all_stats.append({
                            'policy_loss': policy_loss.item(),
                            'baseline': self.baseline.item(),
                            'mean_advantage': per_token_advantage.mean().item(),
                            'num_policy_layers': len(log_probs_list),
                        })

            # Compute regularization (optional)
            if self.config.lambda_compute > 0.0 and all_stats:
                # Average steps across all TTT layers
                avg_steps_list = [s['avg_steps'] for s in all_stats if 'avg_steps' in s]
                if avg_steps_list:
                    avg_steps = sum(avg_steps_list) / len(avg_steps_list)

                    # If target_avg_steps is specified, penalize deviation from target
                    if self.config.target_avg_steps is not None:
                        # Quadratic penalty: (avg_steps - target)^2
                        deviation = avg_steps - self.config.target_avg_steps
                        compute_penalty = self.config.lambda_compute * (deviation ** 2)
                    else:
                        # Linear penalty: just penalize total compute
                        compute_penalty = self.config.lambda_compute * avg_steps

                    total_loss = total_loss + compute_penalty

                    if return_stats:
                        all_stats.append({
                            'lm_loss': lm_loss.item(),
                            'compute_penalty': compute_penalty.item(),
                            'avg_steps_all_layers': avg_steps,
                            'target_avg_steps': self.config.target_avg_steps if self.config.target_avg_steps is not None else 0.0,
                        })

            loss = total_loss

        result = {
            'logits': logits,
            'loss': loss,
        }

        if return_stats and all_stats:
            result['ttt_stats'] = all_stats

        return result

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops_per_token(
        self,
        num_steps: int,
        seq_len: int = 256,
    ) -> float:
        """
        Estimate FLOPs per token for given step count.

        Args:
            num_steps: Number of gradient steps (scalar or per-token average)
            seq_len: Sequence length (for attention complexity)

        Returns:
            Estimated FLOPs per token
        """
        d = self.config.hidden_dim
        d_ffn = self.config.ffn_dim
        num_heads = self.config.num_heads
        head_dim = d // num_heads
        fw_hidden = self.config.fast_weight_hidden_dim
        n = seq_len

        # Standard attention layer FLOPs
        attention_flops = 8 * d * d + 4 * n * d
        ffn_flops = 4 * d * d_ffn
        ln_flops = 4 * d
        standard_layer_flops = attention_flops + ffn_flops + ln_flops

        # Iterative TTT layer FLOPs
        # QKV projections
        qkv_flops = 3 * 2 * d * d

        # Per-step FLOPs (per head)
        step_flops_per_head = (
            2 * (head_dim * fw_hidden + fw_hidden * head_dim) +  # Fast-weight forward
            2 * head_dim * head_dim +                            # Loss
            2 * (head_dim * fw_hidden + fw_hidden * head_dim)    # Backward (gradient)
        )
        step_flops_per_token = num_heads * step_flops_per_head * num_steps

        # Output projection + gate
        out_proj_flops = 2 * d * d
        gate_flops = 2 * d * d
        local_ln_flops = 4 * d

        ttt_layer_flops = (
            qkv_flops +
            step_flops_per_token +
            out_proj_flops +
            gate_flops +
            local_ln_flops +
            ffn_flops +
            ln_flops
        )

        # Total FLOPs
        num_ttt_layers = len(self.config.ttt_layer_indices)
        num_standard_layers = self.config.num_layers - num_ttt_layers

        embedding_flops = d
        lm_head_flops = 2 * d * self.config.vocab_size
        final_ln_flops = 2 * d

        total_flops = (
            embedding_flops +
            num_standard_layers * standard_layer_flops +
            num_ttt_layers * ttt_layer_flops +
            final_ln_flops +
            lm_head_flops
        )

        return float(total_flops)

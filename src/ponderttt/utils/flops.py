"""
Accurate FLOPs counting for TTT models.

Provides detailed FLOP counting for:
- Matrix multiplications (forward and backward)
- Layer normalizations
- Activation functions (GELU, Sigmoid, Softmax)
- Policy networks
- Higher-order gradients (create_graph=True)
"""

from typing import Dict, List, Union
import torch
import torch.nn as nn


class FLOPsCounter:
    """Accurate FLOPs counter for neural network operations."""

    @staticmethod
    def linear_flops(in_features: int, out_features: int, has_bias: bool = False) -> int:
        """
        FLOPs for Linear layer forward pass: y = xW + b

        Args:
            in_features: Input dimension
            out_features: Output dimension
            has_bias: Whether layer has bias term

        Returns:
            Number of FLOPs for forward pass
        """
        # Forward: matmul (2*in*out - out) + bias (out if has_bias)
        forward = 2 * in_features * out_features
        if has_bias:
            forward += out_features
        return forward

    @staticmethod
    def linear_backward_flops(in_features: int, out_features: int, has_bias: bool = False) -> int:
        """
        FLOPs for Linear layer backward pass.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            has_bias: Whether layer has bias term

        Returns:
            Number of FLOPs for backward pass
        """
        # Gradient w.r.t. input: grad_output @ W^T
        grad_input = 2 * in_features * out_features
        # Gradient w.r.t. weight: input^T @ grad_output
        grad_weight = 2 * in_features * out_features
        # Gradient w.r.t. bias: sum(grad_output)
        grad_bias = out_features if has_bias else 0
        return grad_input + grad_weight + grad_bias

    @staticmethod
    def layernorm_flops(normalized_shape: int) -> int:
        """
        FLOPs for LayerNorm.

        Args:
            normalized_shape: Size of normalized dimension

        Returns:
            Number of FLOPs
        """
        n = normalized_shape
        # Mean: sum(x) / n → n + 1
        # Variance: sum((x - mean)^2) / n → 3n + 1
        # Normalize: (x - mean) / sqrt(var + eps) → 2n + 1
        # Scale & shift: x * weight + bias → 2n
        return 8 * n

    @staticmethod
    def gelu_flops(n: int) -> int:
        """
        FLOPs for GELU activation.

        Args:
            n: Number of elements

        Returns:
            Number of FLOPs
        """
        # GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # Approximation: 8 ops per element
        return 8 * n

    @staticmethod
    def sigmoid_flops(n: int) -> int:
        """
        FLOPs for Sigmoid activation.

        Args:
            n: Number of elements

        Returns:
            Number of FLOPs
        """
        # sigmoid(x) = 1 / (1 + exp(-x))
        # Approximation: 4 ops per element (exp + 3 arithmetic)
        return 4 * n

    @staticmethod
    def softmax_flops(n: int, dim_size: int) -> int:
        """
        FLOPs for Softmax.

        Args:
            n: Total number of elements
            dim_size: Size of dimension being normalized

        Returns:
            Number of FLOPs
        """
        # exp(x): n ops
        # sum: dim_size ops
        # divide: n ops
        return 2 * n + dim_size

    @staticmethod
    def mse_loss_flops(n: int) -> int:
        """
        FLOPs for MSE loss.

        Args:
            n: Number of elements

        Returns:
            Number of FLOPs
        """
        # (pred - target)^2: 2n ops
        # mean: 1 op
        return 2 * n + 1

    @staticmethod
    def cross_entropy_flops(batch_size: int, seq_len: int, vocab_size: int) -> int:
        """
        FLOPs for cross-entropy loss.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            vocab_size: Vocabulary size

        Returns:
            Number of FLOPs
        """
        # Softmax: (batch * seq_len * vocab_size)
        # Log: batch * seq_len * vocab_size
        # Gather: batch * seq_len
        # Mean: 1
        n = batch_size * seq_len * vocab_size
        return FLOPsCounter.softmax_flops(n, vocab_size) + n + batch_size * seq_len + 1


class TTTFLOPsAnalyzer:
    """Analyze FLOPs for TTT models with accurate accounting."""

    def __init__(self, config):
        """
        Initialize analyzer.

        Args:
            config: Model configuration object
        """
        self.config = config

    def count_embedding_flops(self, seq_len: int) -> int:
        """
        FLOPs for embedding lookup.

        Args:
            seq_len: Sequence length

        Returns:
            Number of FLOPs
        """
        # Embedding is table lookup (0 FLOPs)
        # But we count addition of token + position embeddings
        return self.config.hidden_dim * seq_len

    def count_attention_flops(self, seq_len: int) -> int:
        """
        FLOPs for standard multi-head attention.

        Args:
            seq_len: Sequence length

        Returns:
            Number of FLOPs
        """
        d = self.config.hidden_dim
        n = seq_len
        num_heads = self.config.num_heads
        head_dim = d // num_heads

        # Q, K, V projections
        qkv_flops = 3 * FLOPsCounter.linear_flops(d, d)

        # Attention scores: Q @ K^T
        # Shape: (batch, num_heads, seq_len, seq_len)
        scores_flops = 2 * num_heads * n * n * head_dim

        # Softmax over attention scores
        softmax_flops = FLOPsCounter.softmax_flops(num_heads * n * n, n)

        # Attention @ V
        attn_v_flops = 2 * num_heads * n * n * head_dim

        # Output projection
        out_flops = FLOPsCounter.linear_flops(d, d)

        # LayerNorm (2x: pre and post)
        ln_flops = 2 * FLOPsCounter.layernorm_flops(d) * n

        total = qkv_flops * n + scores_flops + softmax_flops + attn_v_flops + out_flops * n + ln_flops
        return total

    def count_ttt_iterative_flops(self, seq_len: int, K: Union[int, float]) -> int:
        """
        FLOPs for iterative TTT layer with K gradient steps.

        Includes forward AND backward passes for gradient computation.

        Args:
            seq_len: Sequence length
            K: Average number of gradient steps

        Returns:
            Number of FLOPs
        """
        d = self.config.hidden_dim
        num_heads = self.config.num_heads
        head_dim = d // num_heads

        # Use fast_weight_hidden_dim if available, otherwise default to head_dim * 4
        fw_hidden = getattr(self.config, 'fast_weight_hidden_dim', head_dim * 4)

        # Q, K, V projections
        qkv_flops = 3 * seq_len * FLOPsCounter.linear_flops(d, d)

        # Per-token, per-step operations
        per_token_per_step_flops = 0

        # Forward through fast-weight (MLP: fc1 + fc2)
        per_token_per_step_flops += num_heads * FLOPsCounter.linear_flops(head_dim, fw_hidden)
        per_token_per_step_flops += num_heads * FLOPsCounter.layernorm_flops(fw_hidden)
        per_token_per_step_flops += num_heads * FLOPsCounter.gelu_flops(fw_hidden)
        per_token_per_step_flops += num_heads * FLOPsCounter.linear_flops(fw_hidden, head_dim)

        # Loss computation: MSE(output, target)
        per_token_per_step_flops += num_heads * FLOPsCounter.mse_loss_flops(head_dim)

        # BACKWARD PASS (for gradient computation)
        # Gradient w.r.t. loss → fc2 backward → GELU backward → LayerNorm backward → fc1 backward
        per_token_per_step_flops += num_heads * FLOPsCounter.linear_backward_flops(fw_hidden, head_dim)
        per_token_per_step_flops += num_heads * 2 * fw_hidden  # GELU backward (approx)
        per_token_per_step_flops += num_heads * FLOPsCounter.layernorm_flops(fw_hidden)  # LN backward ≈ forward
        per_token_per_step_flops += num_heads * FLOPsCounter.linear_backward_flops(head_dim, fw_hidden)

        # Parameter update (SGD): param = param - lr * grad
        num_params = num_heads * (head_dim * fw_hidden + fw_hidden + fw_hidden * head_dim + head_dim)
        per_token_per_step_flops += 2 * num_params  # subtract and multiply

        # Total over all tokens and steps
        iterative_total = int(seq_len * K * per_token_per_step_flops)

        # Output computation (after K steps)
        # TTT norm per head per token
        ttt_norm_flops = seq_len * num_heads * FLOPsCounter.layernorm_flops(head_dim)

        # Post norm
        post_norm_flops = seq_len * FLOPsCounter.layernorm_flops(d)

        # Output projection
        out_flops = seq_len * FLOPsCounter.linear_flops(d, d)

        total = qkv_flops + iterative_total + ttt_norm_flops + post_norm_flops + out_flops
        return total

    def count_policy_network_flops(self, seq_len: int) -> int:
        """
        FLOPs for halting policy network.

        Args:
            seq_len: Sequence length

        Returns:
            Number of FLOPs
        """
        d = self.config.hidden_dim

        # Get step_options if available
        if hasattr(self.config, 'step_options'):
            num_steps = len(self.config.step_options)
        else:
            num_steps = 4  # Default: [1, 2, 4, 8]

        # Step predictor MLP
        mlp_flops = seq_len * FLOPsCounter.linear_flops(d, d)
        mlp_flops += seq_len * FLOPsCounter.gelu_flops(d)
        mlp_flops += seq_len * FLOPsCounter.linear_flops(d, num_steps)

        # Softmax (for Gumbel-softmax)
        mlp_flops += seq_len * FLOPsCounter.softmax_flops(num_steps, num_steps)

        return mlp_flops

    def count_ffn_flops(self, seq_len: int) -> int:
        """
        FLOPs for feed-forward network.

        Args:
            seq_len: Sequence length

        Returns:
            Number of FLOPs
        """
        d = self.config.hidden_dim
        d_ffn = getattr(self.config, 'ffn_dim', d * 4)

        # fc1
        flops = seq_len * FLOPsCounter.linear_flops(d, d_ffn)
        # GELU
        flops += seq_len * FLOPsCounter.gelu_flops(d_ffn)
        # fc2
        flops += seq_len * FLOPsCounter.linear_flops(d_ffn, d)
        # LayerNorm
        flops += seq_len * FLOPsCounter.layernorm_flops(d)

        return flops

    def count_lm_head_flops(self, seq_len: int) -> int:
        """
        FLOPs for language model head.

        Args:
            seq_len: Sequence length

        Returns:
            Number of FLOPs
        """
        d = self.config.hidden_dim
        vocab = self.config.vocab_size

        # Final LayerNorm
        ln_flops = seq_len * FLOPsCounter.layernorm_flops(d)

        # Projection to vocab
        proj_flops = seq_len * FLOPsCounter.linear_flops(d, vocab, has_bias=False)

        # Cross-entropy loss (if computing loss)
        loss_flops = FLOPsCounter.cross_entropy_flops(1, seq_len, vocab)

        return ln_flops + proj_flops + loss_flops

    def estimate_total_flops(
        self,
        seq_len: int,
        num_steps: Union[int, float, torch.Tensor],
        include_backward: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate total FLOPs for full forward (and backward) pass.

        Args:
            seq_len: Sequence length
            num_steps: Average number of gradient steps (for iterative TTT)
            include_backward: Whether to include backward pass for LM training

        Returns:
            Dictionary with FLOPs breakdown
        """
        flops: Dict[str, float] = {}

        # Embedding
        flops['embedding'] = float(self.count_embedding_flops(seq_len))

        # Get TTT layer indices
        if hasattr(self.config, 'ttt_layer_indices'):
            num_ttt_layers = len(self.config.ttt_layer_indices)
        else:
            num_ttt_layers = 0

        num_standard_layers = self.config.num_layers - num_ttt_layers

        # Standard attention layers
        if num_standard_layers > 0:
            flops['attention_layers'] = float(num_standard_layers * self.count_attention_flops(seq_len))
        else:
            flops['attention_layers'] = 0.0

        # TTT layers
        if num_ttt_layers > 0:
            if isinstance(num_steps, torch.Tensor):
                avg_steps = num_steps.float().mean().item()
            else:
                avg_steps = float(num_steps)

            ttt_flops = self.count_ttt_iterative_flops(seq_len, avg_steps)
            flops['ttt_layers'] = float(num_ttt_layers * ttt_flops)
        else:
            flops['ttt_layers'] = 0.0

        # Policy network (if learned)
        use_policy = getattr(self.config, 'use_learned_policy', False)
        if use_policy and num_ttt_layers > 0:
            flops['policy_network'] = float(num_ttt_layers * self.count_policy_network_flops(seq_len))
        else:
            flops['policy_network'] = 0.0

        # FFN layers
        flops['ffn_layers'] = float(self.config.num_layers * self.count_ffn_flops(seq_len))

        # LM head
        flops['lm_head'] = float(self.count_lm_head_flops(seq_len))

        # Total forward
        flops['forward_total'] = sum(flops.values())

        # Backward pass (approximate as 2x forward for parameters with gradients)
        if include_backward:
            # Backward through all except embedding
            backward_flops = (flops['forward_total'] - flops['embedding']) * 2
            flops['backward_total'] = backward_flops
            flops['total'] = flops['forward_total'] + backward_flops
        else:
            flops['backward_total'] = 0.0
            flops['total'] = flops['forward_total']

        # Per-token FLOPs
        flops['per_token'] = flops['total'] / seq_len

        return flops

    def compare_configurations(
        self,
        seq_len: int = 256,
        k_values: List[int] = [1, 2, 4, 8],
        include_backward: bool = True,
    ) -> Dict[str, Dict]:
        """
        Compare FLOPs for different K configurations.

        Args:
            seq_len: Sequence length
            k_values: List of K values to compare
            include_backward: Include backward pass

        Returns:
            Dictionary mapping K to FLOPs breakdown
        """
        results = {}

        for k in k_values:
            results[f'K={k}'] = self.estimate_total_flops(
                seq_len=seq_len,
                num_steps=k,
                include_backward=include_backward
            )

        return results

    def print_comparison(
        self,
        comparison: Dict[str, Dict],
        seq_len: int,
    ):
        """
        Print formatted comparison of configurations.

        Args:
            comparison: Results from compare_configurations()
            seq_len: Sequence length used
        """
        print(f"\n{'=' * 80}")
        print(f"FLOPs Comparison (seq_len={seq_len})")
        print(f"{'=' * 80}")
        print(f"{'Config':<15} {'Total (M)':>15} {'Per-token (K)':>18} {'Forward (M)':>15}")
        print('-' * 80)

        for config_name, flops in comparison.items():
            print(
                f"{config_name:<15} "
                f"{flops['total']/1e6:>15.2f} "
                f"{flops['per_token']/1e3:>18.2f} "
                f"{flops['forward_total']/1e6:>15.2f}"
            )

        print('=' * 80)


def compute_model_flops(
    model,
    seq_len: int,
    num_steps: Union[int, torch.Tensor],
    include_backward: bool = True,
) -> Dict[str, float]:
    """
    Compute FLOPs for a model given configuration.

    Args:
        model: Model instance or config
        seq_len: Sequence length
        num_steps: Number of TTT steps (can be tensor for adaptive)
        include_backward: Include backward pass

    Returns:
        Dictionary with FLOPs breakdown
    """
    # Get config from model if needed
    if hasattr(model, 'config'):
        config = model.config
    else:
        config = model

    analyzer = TTTFLOPsAnalyzer(config)
    return analyzer.estimate_total_flops(
        seq_len=seq_len,
        num_steps=num_steps,
        include_backward=include_backward
    )

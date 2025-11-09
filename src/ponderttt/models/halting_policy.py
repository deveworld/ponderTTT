"""
Learned halting policy for adaptive iteration allocation.

Implements a neural network that predicts the optimal number of TTT gradient
steps for each token, enabling end-to-end learning of compute allocation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaltingPolicyNetwork(nn.Module):
    """
    Learned policy network that predicts iteration counts per token.

    Uses Gumbel-Softmax for differentiable discrete decisions, allowing
    the policy to be trained end-to-end via backpropagation.

    Args:
        hidden_dim: Hidden dimension of input representations
        step_options: List of possible step counts (e.g., [1, 2, 4, 8])
        context_window: Size of context window for LSTM (0 = no context)
        use_lstm: Whether to use LSTM for context encoding
        gumbel_tau: Temperature for Gumbel-Softmax (lower = more discrete)
        predict_per_head: Whether to predict steps per attention head
        num_heads: Number of attention heads (if predict_per_head=True)
    """

    def __init__(
        self,
        hidden_dim: int,
        step_options: List[int] = [1, 2, 4, 8],
        context_window: int = 5,
        use_lstm: bool = True,
        gumbel_tau: float = 1.0,
        predict_per_head: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.step_options = step_options
        self.num_options = len(step_options)
        self.context_window = context_window
        self.use_lstm = use_lstm
        self.gumbel_tau = gumbel_tau
        self.predict_per_head = predict_per_head
        self.num_heads = num_heads

        # Register step options as buffer (not a parameter)
        self.register_buffer(
            "step_options_tensor",
            torch.tensor(step_options, dtype=torch.float32)
        )

        # Context encoder
        if use_lstm and context_window > 0:
            # Bidirectional LSTM for capturing local context
            self.context_encoder = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            predictor_input_dim = hidden_dim * 2
        else:
            self.context_encoder = None
            predictor_input_dim = hidden_dim

        # Step predictor
        if predict_per_head:
            # Predict different step counts for each head
            self.step_predictor = nn.Sequential(
                nn.Linear(predictor_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_heads * self.num_options)
            )
        else:
            # Predict single step count for all heads
            self.step_predictor = nn.Sequential(
                nn.Linear(predictor_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, self.num_options)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_probs: bool = False,
        deterministic: bool = False,
        pooling: str = 'none',
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict number of steps for each token using REINFORCE.

        Args:
            hidden_states: Input representations (batch, seq_len, hidden_dim)
            return_probs: Whether to return log probabilities (for REINFORCE)
            deterministic: If True, use argmax instead of sampling (for inference)
            pooling: Batch aggregation strategy: 'none' (per-token), 'mean', 'max'

        Returns:
            steps: Predicted step counts
                - If pooling='none': (batch, seq_len) - true per-token
                - If pooling='mean'/'max': (seq_len,) - per-position
            log_probs: Log probabilities for REINFORCE or None
                - Shape matches steps
        """
        batch_size, seq_len, _ = hidden_states.shape

        if self.context_encoder is not None:
            context, _ = self.context_encoder(hidden_states)
        else:
            context = hidden_states

        step_logits = self.step_predictor(context)

        if self.predict_per_head:
            step_logits = step_logits.view(batch_size, seq_len, self.num_heads, self.num_options)
            step_logits = step_logits.mean(dim=2)

        # Apply batch pooling based on strategy
        if pooling == 'mean':
            step_logits = step_logits.mean(dim=0)
        elif pooling == 'max':
            step_logits, _ = step_logits.max(dim=0)
        elif pooling == 'none':
            pass
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

        if self.training and not deterministic:
            # REINFORCE: Sample from categorical distribution
            dist = torch.distributions.Categorical(logits=step_logits)
            step_indices = dist.sample()
            log_probs = dist.log_prob(step_indices)
            steps = self.step_options_tensor[step_indices].long()

            if return_probs:
                return steps, log_probs
            else:
                return steps, None
        else:
            # Evaluation: Use argmax (deterministic)
            step_indices = torch.argmax(step_logits, dim=-1)
            steps = self.step_options_tensor[step_indices].long()

            if return_probs:
                probs = F.softmax(step_logits, dim=-1)
                return steps, probs
            else:
                return steps, None

    def compute_entropy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute policy entropy (useful for exploration bonus).

        Args:
            hidden_states: Input representations (batch, seq_len, hidden_dim)

        Returns:
            entropy: Policy entropy per token (batch, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Encode context
        if self.context_encoder is not None:
            context, _ = self.context_encoder(hidden_states)
        else:
            context = hidden_states

        # Get logits
        step_logits = self.step_predictor(context)

        if self.predict_per_head:
            step_logits = step_logits.view(batch_size, seq_len, self.num_heads, self.num_options)
            # Average entropy across heads
            probs = F.softmax(step_logits, dim=-1)
            log_probs = F.log_softmax(step_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean(dim=-1)  # (batch, seq_len)
        else:
            probs = F.softmax(step_logits, dim=-1)
            log_probs = F.log_softmax(step_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

        return entropy


class MultiGranularityRouter(nn.Module):
    """
    Hierarchical routing: layer-wise + token-wise step allocation.

    Decides both which layers should use TTT (layer-wise routing) and
    how many steps each token should receive (token-wise halting).

    Args:
        num_layers: Total number of transformer layers
        hidden_dim: Hidden dimension
        step_options: List of possible step counts
        use_layer_routing: Whether to enable layer-wise routing
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        step_options: List[int] = [1, 2, 4, 8],
        use_layer_routing: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.step_options = step_options
        self.use_layer_routing = use_layer_routing

        # Layer-wise router
        if use_layer_routing:
            self.layer_router = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_layers),
            )
        else:
            self.layer_router = None

        # Token-wise step predictors (one per layer)
        self.step_predictors = nn.ModuleList([
            HaltingPolicyNetwork(
                hidden_dim=hidden_dim,
                step_options=step_options,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make routing decision for a specific layer.

        Args:
            hidden_states: Input representations (batch, seq_len, hidden_dim)
            layer_idx: Index of current layer

        Returns:
            use_ttt: Binary decision whether to use TTT (batch,)
            num_steps: Number of steps per token (batch, seq_len)
        """
        batch_size = hidden_states.shape[0]

        # Layer-wise routing decision
        if self.layer_router is not None:
            # Pool across sequence dimension
            pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)

            # Get routing logits for all layers
            layer_logits = self.layer_router(pooled)  # (batch, num_layers)

            # Get decision for this specific layer
            if self.training:
                # Gumbel-Softmax for differentiable binary decision
                layer_probs = torch.sigmoid(layer_logits[:, layer_idx])
                use_ttt = (torch.rand_like(layer_probs) < layer_probs).float()
            else:
                use_ttt = (torch.sigmoid(layer_logits[:, layer_idx]) > 0.5).float()
        else:
            # Always use TTT
            use_ttt = torch.ones(batch_size, device=hidden_states.device)

        # Token-wise step prediction
        num_steps, _ = self.step_predictors[layer_idx](hidden_states)

        # If layer is disabled, set num_steps to 0
        num_steps = num_steps * use_ttt.unsqueeze(1)

        return use_ttt, num_steps

    def get_layer_usage_stats(self, hidden_states_list: List[torch.Tensor]) -> Dict:
        """
        Analyze which layers are being used most.

        Args:
            hidden_states_list: List of hidden states for each layer

        Returns:
            stats: Dictionary with layer usage statistics
        """
        if self.layer_router is None:
            return {"layer_routing": "disabled"}

        layer_usage = []
        for layer_idx, hidden_states in enumerate(hidden_states_list):
            pooled = hidden_states.mean(dim=1)
            layer_logits = self.layer_router(pooled)
            layer_prob = torch.sigmoid(layer_logits[:, layer_idx]).mean().item()
            layer_usage.append(layer_prob)

        return {
            "layer_usage_prob": layer_usage,
            "most_used_layer": int(torch.tensor(layer_usage).argmax()),
            "least_used_layer": int(torch.tensor(layer_usage).argmin()),
        }

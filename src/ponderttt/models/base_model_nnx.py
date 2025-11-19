"""
Base transformer language model with TTT layers in NNX.

Migrated from Linen to NNX, removing HuggingFace Transformers dependency.
Uses native GPT-2 implementation for TPU optimization.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from ponderttt.models.gpt2_nnx import GPT2Config
from ponderttt.models.ttt_layer_nnx import TTTLayer, TTTConfig
from ponderttt.models.lora_layer_nnx import LoRALayer, LoRAConfig


@dataclass
class ModelConfig:
    """Configuration for base model."""

    model_name: str = "gpt2"
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False
    mesh: Optional[jax.sharding.Mesh] = None
    shard_params: bool = False  # Enable parameter sharding for TPU Pods
    fast_weight_type: str = "ttt"  # "ttt" or "lora"


class TTTTransformerLM(nnx.Module):
    """
    Transformer Language Model with Test-Time Training layers.

    Combines:
    - Slow weights (theta_slow): Frozen pretrained transformer (GPT-2)
    - Fast weights (theta_fast): Adaptive TTT layer

    Following the PonderTTT architecture from PLAN.md:
        output = forward(chunk, theta_slow + theta_fast)
    """

    def __init__(
        self,
        gpt2_config: GPT2Config,
        ttt_config: Optional[TTTConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        model_config: ModelConfig = None,
        rngs: nnx.Rngs = None,
        tie_word_embeddings: bool = True
    ):
        """Initialize TTT-augmented model.

        Args:
            gpt2_config: GPT-2 model configuration
            ttt_config: TTT layer configuration (if using TTT)
            lora_config: LoRA configuration (if using LoRA)
            model_config: Base model configuration
            rngs: Random number generators
            tie_word_embeddings: Whether to share embedding weights with LM head
        """
        if model_config is None:
            model_config = ModelConfig()

        self.gpt2_config = gpt2_config
        self.ttt_config = ttt_config
        self.lora_config = lora_config
        self.model_config = model_config
        self.tie_word_embeddings = tie_word_embeddings
        self.fast_weight_type = model_config.fast_weight_type

        # Slow weights: Pretrained GPT-2 (will be frozen during training)
        # Note: We use GPT2Model (without LM head) to get hidden states
        from ponderttt.models.gpt2_nnx import GPT2Model
        self.base_model = GPT2Model(gpt2_config, rngs)

        # Pre-normalization for fast-weight layer (following official TTT-LM pattern)
        # Official implementation: hidden_states_pre_normed = self.seq_norm(hidden_states)
        self.fast_norm = nnx.LayerNorm(gpt2_config.n_embd, epsilon=1e-5, rngs=rngs)

        # Fast weights: TTT layer or LoRA (adaptive, trainable)
        if self.fast_weight_type == "ttt":
            if ttt_config is None:
                raise ValueError("ttt_config required when fast_weight_type='ttt'")
            self.fast_layer = TTTLayer(ttt_config, rngs)
        elif self.fast_weight_type == "lora":
            if lora_config is None:
                raise ValueError("lora_config required when fast_weight_type='lora'")
            self.fast_layer = LoRALayer(lora_config, rngs)
        else:
            raise ValueError(f"Unknown fast_weight_type: {self.fast_weight_type}")

        # LM head with weight tying
        if not tie_word_embeddings:
            self.lm_head = nnx.Linear(
                gpt2_config.n_embd,
                gpt2_config.vocab_size,
                use_bias=False,
                rngs=rngs
            )

    def freeze_base_model(self):
        """Freeze slow weights (pretrained transformer).

        Sets requires_grad=False for all base model parameters.
        Only TTT layer parameters will be trained.
        """
        # In NNX, we can't directly set requires_grad
        # Instead, we'll filter parameters during optimizer creation
        # This method is kept for compatibility
        pass

    def get_trainable_params(self) -> dict:
        """Get only trainable parameters (fast-weight layer).

        Returns:
            Dictionary containing only fast-weight layer parameters
        """
        # Extract only fast-weight layer parameters (TTT or LoRA)
        # In NNX, we use nnx.split to get state, then filter
        graphdef, state = nnx.split(self)

        # Create filtered state with only fast-weight parameters
        trainable_state = {}
        for key, value in state.items():
            if 'fast_layer' in str(key):
                trainable_state[key] = value

        return trainable_state

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        use_ttt: bool = True,
    ) -> dict:
        """
        Forward pass combining slow and fast weights.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (currently unused)
            use_ttt: Whether to apply TTT layer (False for SKIP baseline)

        Returns:
            Dictionary with:
                - logits: Output logits [batch, seq_len, vocab_size]
                - ttt_stats: TTT layer statistics (if use_ttt=True)

        Note:
            Use model.train() / model.eval() to control dropout.
        """
        # Get hidden states from frozen base model
        # Apply stop_gradient to freeze theta_slow (PLAN.md: only theta_fast is trainable)
        hidden_states = jax.lax.stop_gradient(self.base_model(input_ids))

        train_flag = self.training if hasattr(self, "training") else False

        if use_ttt:
            # Following official TTT-LM Block pattern (model.py Line 696-714):
            # 1. Pre-normalization
            # 2. Fast-weight layer (TTT or LoRA)
            # 3. Residual connection (no scaling)

            # Pre-normalize hidden states (official pattern)
            hidden_states_normed = self.fast_norm(hidden_states)

            # Apply fast-weight layer (TTT or LoRA)
            fast_output, fast_stats = self.fast_layer(
                hidden_states_normed,
                mask=attention_mask,
                position_ids=None,
                train=train_flag,
            )

            # Residual connection (official pattern: hidden_states = hidden_states + seq_modeling_output)
            adapted_hidden = hidden_states + fast_output

            # Project adapted hidden states to vocabulary logits with weight tying
            if self.tie_word_embeddings:
                # Use shared embedding weights for LM head
                # Following official TTT-LM-JAX implementation
                embedding_kernel = self.base_model.wte.embedding.value  # [vocab_size, n_embd]
                logits = adapted_hidden @ embedding_kernel.T  # [batch, seq_len, vocab_size]
            else:
                logits = self.lm_head(adapted_hidden)

            return {
                "logits": logits,
                "ttt_stats": fast_stats,  # Works for both TTT and LoRA
            }
        else:
            # SKIP: Project hidden states directly without TTT adaptation
            if self.tie_word_embeddings:
                embedding_kernel = self.base_model.wte.embedding.value
                logits = hidden_states @ embedding_kernel.T
            else:
                logits = self.lm_head(hidden_states)

            return {
                "logits": logits,
                "ttt_stats": None,
            }


def load_ttt_model(
    model_name: str = "gpt2",
    ttt_config: Optional[TTTConfig] = None,
    lora_config: Optional[LoRAConfig] = None,
    fast_weight_type: str = "ttt",
    dtype: jnp.dtype = jnp.float32,
    seed: int = 0,
    load_pretrained: bool = True,
) -> Tuple[TTTTransformerLM, GPT2Config]:
    """
    Load TTT-augmented transformer model.

    Args:
        model_name: GPT-2 variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
        ttt_config: TTT layer configuration (if fast_weight_type='ttt')
        lora_config: LoRA configuration (if fast_weight_type='lora')
        fast_weight_type: Type of fast weights ("ttt" or "lora")
        dtype: Data type for model
        seed: Random seed
        load_pretrained: Whether to load pretrained weights from HuggingFace

    Returns:
        (model, config) tuple
    """
    # Get GPT-2 config
    gpt2_config = GPT2Config.from_pretrained(model_name)

    # Default configs if not provided
    if fast_weight_type == "ttt" and ttt_config is None:
        ttt_config = TTTConfig(
            hidden_dim=gpt2_config.n_embd,  # Match GPT-2 hidden size
            num_heads=gpt2_config.n_head,
            head_dim=gpt2_config.n_embd // gpt2_config.n_head,
            mini_batch_size=16,
            dtype=dtype,
        )
    elif fast_weight_type == "lora" and lora_config is None:
        lora_config = LoRAConfig(
            hidden_dim=gpt2_config.n_embd,
            rank=64,  # Default rank
            alpha=64.0,
            dropout_rate=0.1,
            dtype=dtype,
        )

    # Base model config
    model_config = ModelConfig(
        model_name=model_name,
        dtype=dtype,
        fast_weight_type=fast_weight_type,
    )

    # Create model
    rngs = nnx.Rngs(seed)
    model = TTTTransformerLM(
        gpt2_config=gpt2_config,
        ttt_config=ttt_config,
        lora_config=lora_config,
        model_config=model_config,
        rngs=rngs,
        tie_word_embeddings=True,
    )

    # Load pretrained weights if requested
    if load_pretrained:
        from ponderttt.models.checkpoint_converter import load_huggingface_weights
        from ponderttt.models.gpt2_nnx import GPT2LMHeadModel

        # Create temporary full model to load weights
        temp_model = GPT2LMHeadModel(gpt2_config, rngs, tie_word_embeddings=True)
        temp_model = load_huggingface_weights(temp_model, model_name)

        # Copy weights to our base_model
        # Token embeddings
        model.base_model.wte.embedding.value = temp_model.transformer.wte.embedding.value
        model.base_model.wpe.embedding.value = temp_model.transformer.wpe.embedding.value

        # Transformer blocks
        for i in range(gpt2_config.n_layer):
            src_block = temp_model.transformer.h[i]
            dst_block = model.base_model.h[i]

            # Copy all block parameters
            dst_block.ln_1.scale.value = src_block.ln_1.scale.value
            dst_block.ln_1.bias.value = src_block.ln_1.bias.value

            dst_block.attn.c_attn.kernel.value = src_block.attn.c_attn.kernel.value
            dst_block.attn.c_attn.bias.value = src_block.attn.c_attn.bias.value
            dst_block.attn.c_proj.kernel.value = src_block.attn.c_proj.kernel.value
            dst_block.attn.c_proj.bias.value = src_block.attn.c_proj.bias.value

            dst_block.ln_2.scale.value = src_block.ln_2.scale.value
            dst_block.ln_2.bias.value = src_block.ln_2.bias.value

            dst_block.mlp.c_fc.kernel.value = src_block.mlp.c_fc.kernel.value
            dst_block.mlp.c_fc.bias.value = src_block.mlp.c_fc.bias.value
            dst_block.mlp.c_proj.kernel.value = src_block.mlp.c_proj.kernel.value
            dst_block.mlp.c_proj.bias.value = src_block.mlp.c_proj.bias.value

        # Final layer norm
        model.base_model.ln_f.scale.value = temp_model.transformer.ln_f.scale.value
        model.base_model.ln_f.bias.value = temp_model.transformer.ln_f.bias.value

        print(f"OK Loaded pretrained weights from {model_name}")

    return model, gpt2_config


def count_parameters(model: nnx.Module) -> int:
    """
    Count number of parameters in an NNX model.

    Args:
        model: NNX module

    Returns:
        Total number of parameters
    """
    _, state = nnx.split(model)
    return sum(
        x.size if isinstance(x, jnp.ndarray) else 0
        for x in jax.tree_util.tree_leaves(state)
    )


def count_trainable_parameters(model: TTTTransformerLM) -> Tuple[int, int]:
    """
    Count trainable and total parameters separately.

    Args:
        model: TTT model

    Returns:
        (trainable_params, total_params) tuple
    """
    # Get full state with all nested parameters
    state = nnx.state(model)

    total = 0
    trainable = 0

    # Flatten state to get all parameters with their paths
    flat_params, _ = jax.tree_util.tree_flatten_with_path(state)

    for path, value in flat_params:
        if isinstance(value, jnp.ndarray):
            param_count = value.size
            total += param_count

            # Only fast-weight layer parameters are trainable
            # Path is tuple like (DictKey('fast_layer'), DictKey('q_lora'), ...)
            path_str = '/'.join(str(p) for p in path)
            if 'fast_layer' in path_str:
                trainable += param_count

    return trainable, total

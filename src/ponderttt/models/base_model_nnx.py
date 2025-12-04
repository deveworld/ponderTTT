"""
Base transformer language model with TTT layers in NNX.

Migrated from Linen to NNX, removing HuggingFace Transformers dependency.
Supports both GPT-2 and Gemma 3 (4B, 12B) as base models.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, Union, TYPE_CHECKING, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx

from ponderttt.models.gpt2_nnx import GPT2Config
from ponderttt.models.ttt_layer_nnx import TTTLayer, TTTConfig
from ponderttt.models.lora_layer_nnx import LoRALayer, LoRAConfig

if TYPE_CHECKING:
    from ponderttt.models.gemma3 import Gemma3Config, Gemma3TTTModel


@runtime_checkable
class TTTModelProtocol(Protocol):
    """Protocol defining the common interface for TTT models.

    Both TTTTransformerLM and Gemma3TTTModel implement this interface.
    """
    fast_layer: Union[TTTLayer, LoRALayer]

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        use_ttt: bool = True,
        gating_scale: Optional[jax.Array] = None,
    ) -> dict: ...

    def get_trainable_params(self) -> nnx.State: ...

    def train(self, **attributes: Any) -> None: ...

    def eval(self, **attributes: Any) -> None: ...


# Type alias for any TTT model (GPT-2 based or Gemma 3 based)
TTTModel = Union["TTTTransformerLM", "Gemma3TTTModel"]

# Type alias for any model config
ModelConfigType = Union[GPT2Config, "Gemma3Config"]


@dataclass
class ModelConfig:
    """Configuration for base model."""

    model_name: str = "gpt2"
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False
    mesh: Optional[jax.sharding.Mesh] = None
    shard_params: bool = False  # Enable parameter sharding for TPU Pods
    fast_weight_type: str = "ttt"  # "ttt" or "lora"
    pad_token_id: Optional[int] = None  # Optional pad token id to mask logits


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
        rngs: nnx.Rngs,
        ttt_config: Optional[TTTConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        model_config: Optional[ModelConfig] = None,
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
        self.is_training = False
        self.pad_token_id = model_config.pad_token_id

        # Slow weights: Pretrained GPT-2 (will be frozen during training)
        # Note: We use GPT2Model (without LM head) to get hidden states
        from ponderttt.models.gpt2_nnx import GPT2Model
        self.base_model = GPT2Model(gpt2_config, rngs)

        # Pre-normalization for fast-weight layer (following official TTT-LM pattern)
        # Official implementation: hidden_states_pre_normed = self.seq_norm(hidden_states)
        self.fast_norm = nnx.LayerNorm(gpt2_config.n_embd, epsilon=1e-5, rngs=rngs)

        # Fast weights: TTT layer or LoRA (adaptive, trainable)
        self.fast_layer: Union[TTTLayer, LoRALayer]
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

    def train(self, **attributes):
        super().train(**attributes)
        self.is_training = True

    def eval(self, **attributes):
        super().eval(**attributes)
        self.is_training = False

    def get_trainable_params(self) -> nnx.State:
        """Get only trainable parameters (fast-weight layer).

        Returns:
            nnx.State containing only fast-weight layer parameters
        """
        # Extract only fast-weight layer parameters (TTT or LoRA)
        _, fast_state = nnx.split(self.fast_layer)
        return fast_state

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        use_ttt: bool = True,
        gating_scale: Optional[jax.Array] = None,
    ) -> dict:
        """
        Forward pass combining slow and fast weights.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (currently unused)
            position_ids: Position IDs [batch, seq_len]
            use_ttt: Whether to apply TTT layer (False for SKIP baseline)
            gating_scale: Optional scaling factor for TTT update (for differentiable gating)

        Returns:
            Dictionary with:
                - logits: Output logits [batch, seq_len, vocab_size]
                - ttt_stats: TTT layer statistics (if use_ttt=True)

        Note:
            Use model.train() / model.eval() to control dropout.
        """
        # Get hidden states from frozen base model
        # Apply stop_gradient to freeze theta_slow (PLAN.md: only theta_fast is trainable)
        train_flag = self.is_training
        
        # If position_ids not provided, generate them (0..T)
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0)

        hidden_states = jax.lax.stop_gradient(self.base_model(
            input_ids, 
            position_ids=position_ids,
            train=train_flag
        ))

        if use_ttt:
            # Following official TTT-LM Block pattern (model.py Line 696-714):
            # 1. Pre-normalization
            # 2. Fast-weight layer (TTT or LoRA)
            # 3. Residual connection (no scaling)

            # Pre-normalize hidden states (official pattern)
            hidden_states_normed = self.fast_norm(hidden_states)

            # Apply fast-weight layer (TTT or LoRA)
            if isinstance(self.fast_layer, TTTLayer):
                fast_output, fast_stats = self.fast_layer(
                    hidden_states_normed,
                    mask=attention_mask,
                    position_ids=position_ids,  # Pass correct positions
                    train=train_flag,
                    gating_scale=gating_scale,
                )
            else:
                # LoRA or other layer
                fast_output, fast_stats = self.fast_layer(
                    hidden_states_normed,
                    mask=attention_mask,
                    position_ids=position_ids,
                    train=train_flag,
                )

            # Residual connection (official pattern: hidden_states = hidden_states + seq_modeling_output)
            adapted_hidden = hidden_states + fast_output

            # Project adapted hidden states to vocabulary logits with weight tying
            if self.tie_word_embeddings:
                # Use shared embedding weights for LM head
                # Following official TTT-LM-JAX implementation
                embedding_kernel = self.base_model.wte.embedding[...]  # [vocab_size, n_embd]
                logits = adapted_hidden @ embedding_kernel.T  # [batch, seq_len, vocab_size]
            else:
                logits = self.lm_head(adapted_hidden)

            if self.pad_token_id is not None:
                logits = logits.at[..., self.pad_token_id].set(-1e9)

            return {
                "logits": logits,
                "hidden_states": hidden_states,  # For feature extraction
                "ttt_stats": fast_stats,  # Works for both TTT and LoRA
            }
        else:
            # SKIP: Project hidden states directly without TTT adaptation
            if self.tie_word_embeddings:
                embedding_kernel = self.base_model.wte.embedding[...]
                logits = hidden_states @ embedding_kernel.T
            else:
                logits = self.lm_head(hidden_states)

            # Mask out pad token logits (added vocab entry) to avoid skewing softmax
            if self.pad_token_id is not None:
                logits = logits.at[..., self.pad_token_id].set(-1e9)

            return {"logits": logits, "hidden_states": hidden_states, "ttt_stats": None}


def load_ttt_model(
    model_name: str = "gpt2",
    ttt_config: Optional[TTTConfig] = None,
    lora_config: Optional[LoRAConfig] = None,
    fast_weight_type: str = "ttt",
    dtype: jnp.dtype = jnp.float32,
    seed: int = 0,
    load_pretrained: bool = True,
    vocab_size: int | None = None,
    pad_token_id: int | None = None,
    checkpoint_path: str | None = None,
) -> Tuple[TTTModel, ModelConfigType]:
    """
    Load TTT-augmented transformer model.

    Args:
        model_name: Model identifier. Supported:
            - GPT-2 variants: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
            - Gemma 3: "gemma3-4b", "gemma3-12b", "gemma3-1b" (test)
        ttt_config: TTT layer configuration (if fast_weight_type='ttt')
        lora_config: LoRA configuration (if fast_weight_type='lora')
        fast_weight_type: Type of fast weights ("ttt" or "lora")
        dtype: Data type for model
        seed: Random seed
        load_pretrained: Whether to load pretrained weights
        vocab_size: Optional override for tokenizer vocab size (GPT-2 only)
        pad_token_id: Optional pad token id to explicitly mask in logits (GPT-2 only)
        checkpoint_path: Path to checkpoint (for Gemma 3 Orbax checkpoints)
            For HuggingFace Gemma 3, use format "hf:google/gemma-3-4b-pt"

    Returns:
        (model, config) tuple
    """
    # Handle Gemma 3 models
    if model_name.startswith("gemma3"):
        return _load_gemma3_ttt_model(
            model_name=model_name,
            ttt_config=ttt_config,
            dtype=dtype,
            seed=seed,
            load_pretrained=load_pretrained,
            checkpoint_path=checkpoint_path,
        )

    # Get GPT-2 config
    gpt2_config = GPT2Config.from_pretrained(model_name)
    base_vocab_size = gpt2_config.vocab_size
    if vocab_size is not None:
        if vocab_size < gpt2_config.vocab_size:
            raise ValueError(f"vocab_size override ({vocab_size}) is smaller than base config ({gpt2_config.vocab_size})")
        gpt2_config.vocab_size = vocab_size
        # If vocab was expanded (e.g., to add pad), assume the new last token is pad unless provided
        if pad_token_id is None and vocab_size > base_vocab_size:
            pad_token_id = vocab_size - 1

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
        pad_token_id=pad_token_id,
    )

    # Create model
    rngs = nnx.Rngs(seed)
    model = TTTTransformerLM(
        gpt2_config=gpt2_config,
        rngs=rngs,
        ttt_config=ttt_config,
        lora_config=lora_config,
        model_config=model_config,
        tie_word_embeddings=True,
    )

    def _set_param_value(param: nnx.Param | None, value: Any, name: str) -> None:
        if param is None:
            raise ValueError(f"Parameter '{name}' is not initialized")
        # NNX Variable in-place update (reading uses [...], writing uses .value)
        param.value = value

    def _get_param_value(param: nnx.Param | None, name: str) -> Any:
        if param is None:
            raise ValueError(f"Parameter '{name}' is not initialized")
        return param[...]

    # Load pretrained weights if requested
    if load_pretrained:
        from ponderttt.models.checkpoint_converter import load_huggingface_weights
        from ponderttt.models.gpt2_nnx import GPT2LMHeadModel

        # Create temporary full model to load weights
        temp_model = GPT2LMHeadModel(gpt2_config, rngs, tie_word_embeddings=True)
        temp_model = load_huggingface_weights(temp_model, model_name)

        # Pad embeddings if vocab_size was expanded (e.g., added pad token)
        vocab_diff = gpt2_config.vocab_size - temp_model.transformer.wte.embedding[...].shape[0]
        def _pad_vocab_matrix(x: Any) -> Any:
            if vocab_diff <= 0:
                return x
            if x.ndim == 2:
                return jnp.pad(x, ((0, vocab_diff), (0, 0)))
            return x

        # Copy weights to our base_model
        _set_param_value(
            model.base_model.wte.embedding,
            _pad_vocab_matrix(_get_param_value(temp_model.transformer.wte.embedding, "wte.embedding")),
            "wte.embedding",
        )
        _set_param_value(
            model.base_model.wpe.embedding,
            _get_param_value(temp_model.transformer.wpe.embedding, "wpe.embedding"),
            "wpe.embedding",
        )

        # Transformer blocks
        for i in range(gpt2_config.n_layer):
            src_block = temp_model.transformer.h[i]
            dst_block = model.base_model.h[i]

            _set_param_value(
                dst_block.ln_1.scale,
                _get_param_value(src_block.ln_1.scale, f"h[{i}].ln_1.scale"),
                f"h[{i}].ln_1.scale",
            )
            _set_param_value(
                dst_block.ln_1.bias,
                _get_param_value(src_block.ln_1.bias, f"h[{i}].ln_1.bias"),
                f"h[{i}].ln_1.bias",
            )

            _set_param_value(
                dst_block.attn.c_attn.kernel,
                _get_param_value(src_block.attn.c_attn.kernel, f"h[{i}].attn.c_attn.kernel"),
                f"h[{i}].attn.c_attn.kernel",
            )
            _set_param_value(
                dst_block.attn.c_attn.bias,
                _get_param_value(src_block.attn.c_attn.bias, f"h[{i}].attn.c_attn.bias"),
                f"h[{i}].attn.c_attn.bias",
            )
            _set_param_value(
                dst_block.attn.c_proj.kernel,
                _get_param_value(src_block.attn.c_proj.kernel, f"h[{i}].attn.c_proj.kernel"),
                f"h[{i}].attn.c_proj.kernel",
            )
            _set_param_value(
                dst_block.attn.c_proj.bias,
                _get_param_value(src_block.attn.c_proj.bias, f"h[{i}].attn.c_proj.bias"),
                f"h[{i}].attn.c_proj.bias",
            )

            _set_param_value(
                dst_block.ln_2.scale,
                _get_param_value(src_block.ln_2.scale, f"h[{i}].ln_2.scale"),
                f"h[{i}].ln_2.scale",
            )
            _set_param_value(
                dst_block.ln_2.bias,
                _get_param_value(src_block.ln_2.bias, f"h[{i}].ln_2.bias"),
                f"h[{i}].ln_2.bias",
            )

            _set_param_value(
                dst_block.mlp.c_fc.kernel,
                _get_param_value(src_block.mlp.c_fc.kernel, f"h[{i}].mlp.c_fc.kernel"),
                f"h[{i}].mlp.c_fc.kernel",
            )
            _set_param_value(
                dst_block.mlp.c_fc.bias,
                _get_param_value(src_block.mlp.c_fc.bias, f"h[{i}].mlp.c_fc.bias"),
                f"h[{i}].mlp.c_fc.bias",
            )
            _set_param_value(
                dst_block.mlp.c_proj.kernel,
                _get_param_value(src_block.mlp.c_proj.kernel, f"h[{i}].mlp.c_proj.kernel"),
                f"h[{i}].mlp.c_proj.kernel",
            )
            _set_param_value(
                dst_block.mlp.c_proj.bias,
                _get_param_value(src_block.mlp.c_proj.bias, f"h[{i}].mlp.c_proj.bias"),
                f"h[{i}].mlp.c_proj.bias",
            )

        # Final layer norm
        _set_param_value(
            model.base_model.ln_f.scale,
            _get_param_value(temp_model.transformer.ln_f.scale, "ln_f.scale"),
            "ln_f.scale",
        )
        _set_param_value(
            model.base_model.ln_f.bias,
            _get_param_value(temp_model.transformer.ln_f.bias, "ln_f.bias"),
            "ln_f.bias",
        )

        print(f"OK Loaded pretrained weights from {model_name}")

    return model, gpt2_config


def _load_gemma3_ttt_model(
    model_name: str,
    ttt_config: Optional[TTTConfig] = None,
    dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 0,
    load_pretrained: bool = True,
    checkpoint_path: str | None = None,
) -> Tuple["Gemma3TTTModel", "Gemma3Config"]:
    """
    Load Gemma 3 TTT model.

    Internal helper for load_ttt_model().

    Args:
        model_name: "gemma3-4b", "gemma3-12b", or "gemma3-1b"
        ttt_config: TTT layer configuration
        dtype: Data type (default bfloat16 for Gemma 3)
        seed: Random seed
        load_pretrained: Whether to load pretrained weights
        checkpoint_path: Path to checkpoint
            - Orbax: "/path/to/checkpoint"
            - HuggingFace: "hf:google/gemma-3-4b-pt"

    Returns:
        (model, config) tuple
    """
    from ponderttt.models.gemma3 import (
        Gemma3Config,
        Gemma3TTTModel,
        load_gemma3_from_orbax,
        load_gemma3_from_huggingface,
    )

    # Select config based on model size
    if "4b" in model_name:
        gemma_config = Gemma3Config.gemma3_4b(dtype=dtype)
    elif "12b" in model_name:
        gemma_config = Gemma3Config.gemma3_12b(dtype=dtype)
    elif "1b" in model_name:
        gemma_config = Gemma3Config.gemma3_1b(dtype=dtype)
    else:
        raise ValueError(
            f"Unknown Gemma 3 model: {model_name}. "
            f"Supported: gemma3-4b, gemma3-12b, gemma3-1b"
        )

    # Default TTT config for Gemma 3
    if ttt_config is None:
        if "4b" in model_name:
            ttt_config = TTTConfig.for_gemma3_4b(dtype=dtype)
        elif "12b" in model_name:
            ttt_config = TTTConfig.for_gemma3_12b(dtype=dtype)
        else:
            # 1B (testing)
            ttt_config = TTTConfig(
                hidden_dim=gemma_config.embed_dim,
                num_heads=gemma_config.num_heads,
                head_dim=gemma_config.head_dim,
                dtype=dtype,
                mini_batch_size=16,
            )

    # Create model
    rngs = nnx.Rngs(seed)
    model = Gemma3TTTModel(
        gemma_config=gemma_config,
        ttt_config=ttt_config,
        rngs=rngs,
        tie_word_embeddings=True,
    )

    # Load pretrained weights
    if load_pretrained and checkpoint_path:
        if checkpoint_path.startswith("hf:"):
            # HuggingFace format: "hf:google/gemma-3-4b-pt"
            hf_model_id = checkpoint_path[3:]  # Remove "hf:" prefix
            model = load_gemma3_from_huggingface(model, hf_model_id)
        else:
            # Orbax checkpoint
            model = load_gemma3_from_orbax(model, checkpoint_path)

    print(f"Created Gemma 3 TTT model: {model_name}")
    print(f"  - Layers: {gemma_config.num_layers}")
    print(f"  - Hidden dim: {gemma_config.embed_dim}")
    print(f"  - Heads: {gemma_config.num_heads} (KV: {gemma_config.num_kv_heads})")
    print(f"  - Head dim: {gemma_config.head_dim}")

    return model, gemma_config


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
    fast_state = model.get_trainable_params()

    total = 0
    trainable = 0

    # Flatten state to get all parameters with their paths
    for value in jax.tree_util.tree_leaves(state):
        if isinstance(value, jnp.ndarray):
            total += int(value.size)

    for value in jax.tree_util.tree_leaves(fast_state):
        if isinstance(value, jnp.ndarray):
            trainable += int(value.size)

    return trainable, total

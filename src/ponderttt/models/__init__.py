"""
Flax NNX models for PonderTTT.

All models now use Flax NNX instead of Linen.
"""

# NNX implementations (primary)
from .base_model_nnx import (
    ModelConfig,
    ModelConfigType,
    TTTModel,
    TTTModelProtocol,
    TTTTransformerLM,
    count_parameters,
    count_trainable_parameters,
    load_ttt_model,
)
from .gpt2_nnx import GPT2Config, GPT2LMHeadModel, GPT2Model, load_gpt2_model
from .lora_layer_nnx import LoRAConfig, LoRALayer, count_lora_parameters
from .ttt_layer_nnx import TTTConfig, TTTLayer
from .gating_nnx import GatingConfig, GatingNetwork

# Legacy Linen implementations (deprecated, kept for compatibility)
# These will be removed in future versions
# Use the NNX versions above instead

__all__ = [
    # NNX models (use these)
    "TTTTransformerLM",
    "TTTModel",
    "TTTModelProtocol",
    "ModelConfig",
    "ModelConfigType",
    "load_ttt_model",
    "count_parameters",
    "count_trainable_parameters",
    "TTTLayer",
    "TTTConfig",
    "LoRALayer",
    "LoRAConfig",
    "count_lora_parameters",
    "GPT2Model",
    "GPT2LMHeadModel",
    "GPT2Config",
    "load_gpt2_model",
    "GatingNetwork",
    "GatingConfig",
]
"""Checkpoint loading utilities for Gemma 3 models.

Supports loading weights from:
1. Orbax checkpoints (official Google format)
2. HuggingFace transformers checkpoints
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx

from .params import load_params, nest_params

if TYPE_CHECKING:
    from .model import Gemma3TTTModel

# Generic type for model to preserve type through loading
T = TypeVar("T", bound=nnx.Module)


Params = Mapping[str, Any]


def load_and_format_params(path: str) -> Params:
    """Load parameters from Orbax checkpoint and format for NNX.

    Args:
        path: Path to Orbax checkpoint directory

    Returns:
        Nested parameter dictionary
    """
    params = load_params(path)
    param_state = jax.tree.map(jnp.array, params)
    remapped_params = param_remapper(param_state)
    nested_params = nest_params(remapped_params)
    return nested_params


def param_remapper(orig_params: Params) -> Params:
    """Remap params to NNX module layout.

    The official checkpoint has 'mlp/' prefix that needs to be remapped.

    Args:
        orig_params: Original dict of parameters

    Returns:
        Dict of params with remapped names
    """
    new_params = {}
    for k, v in orig_params.items():
        if "mlp/" in k:
            layer_name, param = k.rsplit("/", maxsplit=1)
            if layer_name not in new_params:
                new_params[layer_name] = {}
            if "w" in v:
                new_params[layer_name][param] = v["w"]
        else:
            new_params[k] = v
    return new_params


def _map_linen_var_names(key: tuple[str, ...]) -> tuple[str | int, ...]:
    """Map Linen variable names to NNX variable names."""
    new_key = []
    for k in key:
        if k.startswith("layer_"):
            prefix, suffix = k.split("layer_")
            assert not prefix, prefix
            new_key.append("layers")
            new_key.append(int(suffix))
        elif k == "gating_einsum":
            new_key.append("gate_proj")
            new_key.append("kernel")
        elif k == "linear":
            new_key.append("down_proj")
            new_key.append("kernel")
        else:
            new_key.append(k)
    return tuple(new_key)


def load_gemma3_from_orbax(
    model: T,
    checkpoint_path: str,
    transpose_gating_einsum: bool = True,
) -> T:
    """Load Gemma 3 weights from Orbax checkpoint.

    Args:
        model: Initialized NNX Gemma 3 model
        checkpoint_path: Path to Orbax checkpoint
        transpose_gating_einsum: Whether to transpose gating einsum weights

    Returns:
        Model with loaded weights (same type as input)
    """
    params = load_and_format_params(checkpoint_path)

    # Get model state as flat dict with paths
    state = nnx.state(model)
    flat_state = {}

    def flatten_state(obj, prefix=()):
        if isinstance(obj, (dict, nnx.State)):
            for k, v in obj.items():
                flatten_state(v, prefix + (k,))
        elif hasattr(obj, "value"):
            flat_state[prefix] = obj
        else:
            flat_state[prefix] = obj

    flatten_state(state)

    # Assign weights from checkpoint
    def assign_params(nested_params, prefix=()):
        if isinstance(nested_params, dict):
            for k, v in nested_params.items():
                assign_params(v, prefix + (k,))
        else:
            # Map to NNX path
            mapped_path = _map_linen_var_names(prefix)
            val = nested_params

            # Handle gating einsum transpose
            if "gate_proj" in mapped_path:
                if transpose_gating_einsum:
                    val = jnp.swapaxes(val, 1, 2)
                # Split gate and up projections
                if mapped_path in flat_state:
                    flat_state[mapped_path].value = val[0]
                up_path = mapped_path[:-2] + ("up_proj", "kernel")
                if up_path in flat_state:
                    flat_state[up_path].value = val[1]
            elif mapped_path in flat_state:
                flat_state[mapped_path].value = val

    assign_params(params.get("transformer", params))

    # Update model with new state
    nnx.update(model, state)

    print(f"Loaded Gemma 3 weights from {checkpoint_path}")
    return model


def load_gemma3_from_huggingface(
    model: T,
    model_id: str,
    device: str = "cpu",
) -> T:
    """Load Gemma 3 weights from HuggingFace checkpoint.

    Args:
        model: Initialized NNX Gemma 3 model
        model_id: HuggingFace model ID (e.g., "google/gemma-3-4b-pt")
        device: Device to load weights on

    Returns:
        Model with loaded weights (same type as input)

    Note:
        Requires `transformers` and `torch` packages.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            "Loading from HuggingFace requires 'transformers' and 'torch'. "
            "Install with: pip install transformers torch"
        ) from e

    print(f"Loading HuggingFace model: {model_id}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Get gemma_config from model (Gemma3TTTModel has this attribute)
    gemma_config = getattr(model, "gemma_config", None)
    if gemma_config is None:
        raise ValueError("Model must have a 'gemma_config' attribute")

    # Map HuggingFace weights to NNX model
    hf_state_dict = hf_model.state_dict()

    def hf_to_jax(tensor):
        """Convert PyTorch tensor to JAX array."""
        return jnp.array(tensor.cpu().numpy())

    # Weight mapping: HuggingFace -> NNX
    # Note: HuggingFace uses (out, in) for Linear, NNX uses (in, out)
    weight_mapping = {
        # Embeddings
        "model.embed_tokens.weight": ("base_model", "embedder", "input_embedding"),
        # Final norm
        "model.norm.weight": ("base_model", "final_norm", "scale"),
    }

    # Layer weights
    for i in range(gemma_config.num_layers):
        layer_prefix = f"model.layers.{i}"
        nnx_layer = ("base_model", "layers", i)

        layer_mapping = {
            # Attention
            f"{layer_prefix}.self_attn.q_proj.weight": (*nnx_layer, "attn", "q_einsum", "w"),
            f"{layer_prefix}.self_attn.k_proj.weight": (*nnx_layer, "attn", "kv_einsum", "w"),
            f"{layer_prefix}.self_attn.v_proj.weight": (*nnx_layer, "attn", "kv_einsum", "w"),
            f"{layer_prefix}.self_attn.o_proj.weight": (*nnx_layer, "attn", "attn_vec_einsum", "w"),
            # Norms
            f"{layer_prefix}.input_layernorm.weight": (*nnx_layer, "pre_attention_norm", "scale"),
            f"{layer_prefix}.post_attention_layernorm.weight": (*nnx_layer, "pre_ffw_norm", "scale"),
            # MLP
            f"{layer_prefix}.mlp.gate_proj.weight": (*nnx_layer, "mlp", "gate_proj", "kernel"),
            f"{layer_prefix}.mlp.up_proj.weight": (*nnx_layer, "mlp", "up_proj", "kernel"),
            f"{layer_prefix}.mlp.down_proj.weight": (*nnx_layer, "mlp", "down_proj", "kernel"),
        }

        # QK norm (Gemma 3 specific)
        if getattr(gemma_config, "use_qk_norm", False):
            layer_mapping[f"{layer_prefix}.self_attn.q_norm.weight"] = (
                *nnx_layer, "attn", "_query_norm", "scale"
            )
            layer_mapping[f"{layer_prefix}.self_attn.k_norm.weight"] = (
                *nnx_layer, "attn", "_key_norm", "scale"
            )

        # Post norms (Gemma 2/3 specific)
        if getattr(gemma_config, "use_post_attn_norm", False):
            layer_mapping[f"{layer_prefix}.post_attention_norm.weight"] = (
                *nnx_layer, "post_attention_norm", "scale"
            )
        if getattr(gemma_config, "use_post_ffw_norm", False):
            layer_mapping[f"{layer_prefix}.post_ffw_norm.weight"] = (
                *nnx_layer, "post_ffw_norm", "scale"
            )

        weight_mapping.update(layer_mapping)

    # Apply weights
    def get_nested(obj, path):
        for key in path:
            if isinstance(key, int):
                obj = obj[key]
            else:
                obj = getattr(obj, key) if hasattr(obj, key) else obj[key]
        return obj

    def set_nested(obj, path, value):
        for key in path[:-1]:
            if isinstance(key, int):
                obj = obj[key]
            else:
                obj = getattr(obj, key) if hasattr(obj, key) else obj[key]
        final_key = path[-1]
        if hasattr(obj, final_key):
            target = getattr(obj, final_key)
            if hasattr(target, "value"):
                target.value = value
        elif isinstance(obj, dict):
            obj[final_key] = value

    loaded_count = 0
    for hf_name, nnx_path in weight_mapping.items():
        if hf_name in hf_state_dict:
            try:
                weight = hf_to_jax(hf_state_dict[hf_name])

                # Transpose linear weights (HuggingFace: [out, in] -> NNX: [in, out])
                if "weight" in hf_name and weight.ndim == 2:
                    weight = weight.T

                set_nested(model, nnx_path, weight)
                loaded_count += 1
            except Exception as e:
                print(f"Warning: Failed to load {hf_name}: {e}")

    print(f"Loaded {loaded_count} weights from HuggingFace")

    # Clean up
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def save_gemma3_checkpoint(
    model: nnx.Module,
    checkpoint_path: str,
    step: int = 0,
    metadata: dict | None = None,
) -> None:
    """Save Gemma 3 model checkpoint using Orbax.

    Args:
        model: NNX Gemma 3 model to save
        checkpoint_path: Path to save checkpoint
        step: Training step number
        metadata: Optional metadata dict
    """
    from ...utils.checkpointing import save_checkpoint

    state = nnx.state(model)
    save_checkpoint(
        checkpoint_dir=checkpoint_path,
        step=step,
        state={"model": state},
        metadata=metadata or {},
    )
    print(f"Saved Gemma 3 checkpoint to {checkpoint_path}")

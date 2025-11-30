"""Checkpoint loading utilities for Gemma 3 models.

Supports loading weights from:
1. Orbax checkpoints (official Google format)
2. HuggingFace safetensors checkpoints (without torch/transformers)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

from .params import load_params, nest_params

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


def _download_safetensors(
    model_id: str,
    cache_dir: Optional[Path] = None,
) -> dict[str, np.ndarray]:
    """Download and load safetensors from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g., "google/gemma-3-1b-pt")
        cache_dir: Optional cache directory

    Returns:
        Dictionary of numpy arrays from safetensors
    """
    # Try single file first
    try:
        file_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            cache_dir=cache_dir,
        )
        print(f"Loading weights from {file_path}...")
        return load_file(file_path)
    except Exception:
        pass

    # Try sharded safetensors (model-00001-of-00002.safetensors, etc.)
    try:
        # Download the index file first
        index_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            cache_dir=cache_dir,
        )
        import json
        with open(index_path) as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = sorted(set(index.get("weight_map", {}).values()))
        if not shard_files:
            raise ValueError("No shard files found in index")

        # Download and merge all shards
        state_dict = {}
        for shard_file in shard_files:
            shard_path = hf_hub_download(
                repo_id=model_id,
                filename=shard_file,
                cache_dir=cache_dir,
            )
            print(f"Loading shard: {shard_file}")
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)

        return state_dict
    except Exception as e:
        raise RuntimeError(
            f"Could not find safetensors weights for {model_id}. "
            f"Tried 'model.safetensors' and sharded format.\n"
            f"Error: {e}"
        )


def load_gemma3_from_huggingface(
    model: T,
    model_id: str,
    cache_dir: Optional[Path] = None,
) -> T:
    """Load Gemma 3 weights from HuggingFace checkpoint using safetensors.

    Args:
        model: Initialized NNX Gemma 3 model
        model_id: HuggingFace model ID (e.g., "google/gemma-3-1b-pt")
        cache_dir: Optional cache directory for downloaded weights

    Returns:
        Model with loaded weights (same type as input)

    Note:
        Uses safetensors and huggingface_hub directly without torch/transformers.
    """
    print(f"Downloading/Loading weights for {model_id} (safetensors)...")
    state_dict = _download_safetensors(model_id, cache_dir)

    # Get gemma_config from model (Gemma3TTTModel has this attribute)
    gemma_config = getattr(model, "gemma_config", None)
    if gemma_config is None:
        raise ValueError("Model must have a 'gemma_config' attribute")

    def np_to_jax(arr: np.ndarray) -> jnp.ndarray:
        """Convert numpy array to JAX array."""
        return jnp.array(arr)

    # Weight mapping: HuggingFace -> NNX
    # Note: HuggingFace uses (out, in) for Linear, NNX uses (in, out)
    weight_mapping: dict[str, tuple[str | int, ...]] = {
        # Embeddings
        "model.embed_tokens.weight": ("base_model", "embedder", "input_embedding"),
        # Final norm
        "model.norm.weight": ("base_model", "final_norm", "scale"),
    }

    # Layer weights
    for i in range(gemma_config.num_layers):
        layer_prefix = f"model.layers.{i}"
        nnx_layer: tuple[str | int, ...] = ("base_model", "layers", i)

        layer_mapping: dict[str, tuple[str | int, ...]] = {
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
    def set_nested(obj: Any, path: tuple[str | int, ...], value: jnp.ndarray) -> None:
        for key in path[:-1]:
            if isinstance(key, int):
                obj = obj[key]
            else:
                obj = getattr(obj, key) if hasattr(obj, key) else obj[key]
        final_key = path[-1]
        if isinstance(final_key, int):
            obj[final_key] = value
        elif hasattr(obj, final_key):
            target = getattr(obj, final_key)
            if hasattr(target, "value"):
                target.value = value
        elif isinstance(obj, dict):
            obj[final_key] = value

    loaded_count = 0
    for hf_name, nnx_path in weight_mapping.items():
        if hf_name in state_dict:
            try:
                weight = np_to_jax(state_dict[hf_name])

                # Transpose linear weights (HuggingFace: [out, in] -> NNX: [in, out])
                if "weight" in hf_name and weight.ndim == 2:
                    weight = weight.T

                set_nested(model, nnx_path, weight)
                loaded_count += 1
            except Exception as e:
                print(f"Warning: Failed to load {hf_name}: {e}")

    print(f"Loaded {loaded_count} weights from HuggingFace (safetensors)")
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

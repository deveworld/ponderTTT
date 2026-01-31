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

    # Filter to only include array-like values (skip metadata strings etc.)
    def safe_to_array(x):
        """Convert to jnp.array only if it's a valid array-like type."""
        if isinstance(x, (jnp.ndarray, jax.Array)):
            return x
        if hasattr(x, "__array__"):
            # numpy arrays or array-like objects
            return jnp.array(x)
        if isinstance(x, (str, bool, type(None))):
            # Skip metadata values - these will be filtered out later
            return None
        if isinstance(x, (int, float)):
            # Allow scalars
            return jnp.array(x)
        # Unknown type - try to convert, but may fail
        try:
            return jnp.array(x)
        except (TypeError, ValueError):
            return None

    param_state = jax.tree.map(
        safe_to_array, params, is_leaf=lambda x: isinstance(x, str)
    )

    # Filter out None values (non-array metadata)
    def filter_nones(d):
        if isinstance(d, dict):
            return {
                k: filter_nones(v) for k, v in d.items() if filter_nones(v) is not None
            }
        return d

    param_state = filter_nones(param_state)
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

    # Gemma 3 HuggingFace checkpoints use 'language_model.' prefix on all keys
    # Strip this prefix to match our weight_mapping
    prefix = "language_model."
    if any(k.startswith(prefix) for k in state_dict.keys()):
        state_dict = {
            k[len(prefix) :] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }

    # Get gemma_config from model (Gemma3TTTModel has this attribute)
    gemma_config = getattr(model, "gemma_config", None)
    if gemma_config is None:
        raise ValueError("Model must have a 'gemma_config' attribute")

    # Check if model uses GQA (Grouped Query Attention)
    use_gqa = gemma_config.num_heads != gemma_config.num_kv_heads

    def np_to_jax(arr: np.ndarray) -> jnp.ndarray:
        """Convert numpy array to JAX array."""
        return jnp.array(arr)

    # Weight mapping: HuggingFace -> NNX
    # Note: HuggingFace uses (out, in) for Linear, NNX uses (in, out)
    # For einsum layers, reshape is also needed.
    weight_mapping: dict[str, tuple[str | int, ...]] = {
        # Embeddings
        "model.embed_tokens.weight": ("base_model", "embedder", "input_embedding"),
        # Final norm
        "model.norm.weight": ("base_model", "final_norm", "scale"),
    }

    # Special handling for attention Q/K/V weights (handled separately for GQA)
    # These are NOT added to weight_mapping for GQA models
    attn_qkv_keys: list[tuple[int, str, str]] = []  # (layer_idx, hf_key, qkv_type)

    # Layer weights
    for i in range(gemma_config.num_layers):
        layer_prefix = f"model.layers.{i}"
        nnx_layer: tuple[str | int, ...] = ("base_model", "layers", i)

        # For GQA models, Q/K/V need special handling
        if use_gqa:
            attn_qkv_keys.append((i, f"{layer_prefix}.self_attn.q_proj.weight", "q"))
            attn_qkv_keys.append((i, f"{layer_prefix}.self_attn.k_proj.weight", "k"))
            attn_qkv_keys.append((i, f"{layer_prefix}.self_attn.v_proj.weight", "v"))
            layer_mapping: dict[str, tuple[str | int, ...]] = {
                # Output projection
                f"{layer_prefix}.self_attn.o_proj.weight": (
                    *nnx_layer,
                    "attn",
                    "attn_vec_einsum",
                    "w",
                ),
            }
        else:
            # MHA: use qkv_einsum (handled separately too)
            attn_qkv_keys.append((i, f"{layer_prefix}.self_attn.q_proj.weight", "q"))
            attn_qkv_keys.append((i, f"{layer_prefix}.self_attn.k_proj.weight", "k"))
            attn_qkv_keys.append((i, f"{layer_prefix}.self_attn.v_proj.weight", "v"))
            layer_mapping = {
                f"{layer_prefix}.self_attn.o_proj.weight": (
                    *nnx_layer,
                    "attn",
                    "attn_vec_einsum",
                    "w",
                ),
            }

        # Norms - Gemma 3 HuggingFace naming:
        #   input_layernorm = before attention
        #   post_attention_layernorm = after attention, before FFN residual (this IS the pre-FFN norm!)
        #   pre_feedforward_layernorm = actually used as the norm before FFN in some configs
        #   post_feedforward_layernorm = after FFN
        layer_mapping[f"{layer_prefix}.input_layernorm.weight"] = (
            *nnx_layer,
            "pre_attention_norm",
            "scale",
        )
        # post_attention_layernorm in HF is post_attention_norm in NNX (scale after attention)
        layer_mapping[f"{layer_prefix}.post_attention_layernorm.weight"] = (
            *nnx_layer,
            "post_attention_norm",
            "scale",
        )
        # pre_feedforward_layernorm in HF is pre_ffw_norm in NNX (scale before FFN)
        layer_mapping[f"{layer_prefix}.pre_feedforward_layernorm.weight"] = (
            *nnx_layer,
            "pre_ffw_norm",
            "scale",
        )

        # MLP
        layer_mapping[f"{layer_prefix}.mlp.gate_proj.weight"] = (
            *nnx_layer,
            "mlp",
            "gate_proj",
            "kernel",
        )
        layer_mapping[f"{layer_prefix}.mlp.up_proj.weight"] = (
            *nnx_layer,
            "mlp",
            "up_proj",
            "kernel",
        )
        layer_mapping[f"{layer_prefix}.mlp.down_proj.weight"] = (
            *nnx_layer,
            "mlp",
            "down_proj",
            "kernel",
        )

        # QK norm (Gemma 3 specific)
        if getattr(gemma_config, "use_qk_norm", False):
            layer_mapping[f"{layer_prefix}.self_attn.q_norm.weight"] = (
                *nnx_layer,
                "attn",
                "_query_norm",
                "scale",
            )
            layer_mapping[f"{layer_prefix}.self_attn.k_norm.weight"] = (
                *nnx_layer,
                "attn",
                "_key_norm",
                "scale",
            )

            # Additional Gemma 3 norms (pre/post feedforward norms)
            # These are used in Gemma 3 for additional normalization
            # This case (use_post_attn_norm) is now handled above in the main block
            pass
        if getattr(gemma_config, "use_post_ffw_norm", False):
            # post_feedforward_layernorm in HF = post_ffw_norm in NNX
            layer_mapping[f"{layer_prefix}.post_feedforward_layernorm.weight"] = (
                *nnx_layer,
                "post_ffw_norm",
                "scale",
            )

        weight_mapping.update(layer_mapping)

    # Apply weights
    def set_nested(
        obj: Any, path: tuple[str | int, ...], value: jnp.ndarray, debug_name: str = ""
    ) -> bool:
        """Returns True if successfully set, False otherwise."""
        try:
            for key in path[:-1]:
                if isinstance(key, int):
                    obj = obj[key]
                else:
                    if hasattr(obj, key):
                        obj = getattr(obj, key)
                    elif isinstance(obj, dict) and key in obj:
                        obj = obj[key]
                    else:
                        print(f"Path traversal failed at '{key}' for {debug_name}")
                        return False
            final_key = path[-1]
            if isinstance(final_key, int):
                obj[final_key] = value
                return True
            elif hasattr(obj, final_key):
                target = getattr(obj, final_key)
                if hasattr(target, "value"):
                    target.value = value
                    return True
                else:
                    print(f"Target {final_key} has no 'value' attr for {debug_name}")
                    return False
            elif isinstance(obj, dict):
                obj[final_key] = value
                return True
            else:
                print(f"Cannot set final_key '{final_key}' for {debug_name}")
                return False
        except Exception as e:
            print(f"Exception in set_nested for {debug_name}: {e}")
            return False

    def get_nested(obj: Any, path: tuple[str | int, ...]) -> Any:
        for key in path:
            if isinstance(key, int):
                obj = obj[key]
            else:
                obj = getattr(obj, key) if hasattr(obj, key) else obj[key]
        return obj

    loaded_count = 0

    # Handle Q/K/V attention weights specially
    # Group by layer
    qkv_by_layer: dict[int, dict[str, jnp.ndarray]] = {}
    for layer_idx, hf_key, qkv_type in attn_qkv_keys:
        if hf_key in state_dict:
            weight = np_to_jax(state_dict[hf_key])
            # HuggingFace: [out_features, in_features] -> transpose to [in, out]
            weight = weight.T  # Now [embed_dim, num_heads * head_dim] or [embed_dim, num_kv_heads * head_dim]
            if layer_idx not in qkv_by_layer:
                qkv_by_layer[layer_idx] = {}
            qkv_by_layer[layer_idx][qkv_type] = weight
            loaded_count += 1

    # Now assign Q/K/V weights with proper reshaping
    for layer_idx, qkv_weights in qkv_by_layer.items():
        nnx_layer = ("base_model", "layers", layer_idx, "attn")

        if use_gqa:
            # GQA: separate q_einsum and kv_einsum
            # q_einsum.w: [num_heads, features, head_dim]
            # kv_einsum.w: [2, num_kv_heads, features, head_dim]
            if "q" in qkv_weights:
                q_weight = qkv_weights["q"]  # [embed_dim, num_heads * head_dim]
                # Reshape to [embed_dim, num_heads, head_dim] then transpose to [num_heads, embed_dim, head_dim]
                q_weight = q_weight.reshape(
                    gemma_config.embed_dim,
                    gemma_config.num_heads,
                    gemma_config.head_dim,
                )
                q_weight = jnp.transpose(
                    q_weight, (1, 0, 2)
                )  # [num_heads, embed_dim, head_dim]
                success = set_nested(
                    model,
                    (*nnx_layer, "q_einsum", "w"),
                    q_weight,
                    f"layer{layer_idx}.q_einsum",
                )
                if not success:
                    print(f"Failed to set q_einsum for layer {layer_idx}")

            if "k" in qkv_weights and "v" in qkv_weights:
                k_weight = qkv_weights["k"]  # [embed_dim, num_kv_heads * head_dim]
                v_weight = qkv_weights["v"]  # [embed_dim, num_kv_heads * head_dim]
                # Reshape each to [embed_dim, num_kv_heads, head_dim]
                k_weight = k_weight.reshape(
                    gemma_config.embed_dim,
                    gemma_config.num_kv_heads,
                    gemma_config.head_dim,
                )
                v_weight = v_weight.reshape(
                    gemma_config.embed_dim,
                    gemma_config.num_kv_heads,
                    gemma_config.head_dim,
                )
                # Stack to [2, embed_dim, num_kv_heads, head_dim]
                kv_weight = jnp.stack([k_weight, v_weight], axis=0)
                # Transpose to [2, num_kv_heads, embed_dim, head_dim]
                kv_weight = jnp.transpose(kv_weight, (0, 2, 1, 3))
                success = set_nested(
                    model,
                    (*nnx_layer, "kv_einsum", "w"),
                    kv_weight,
                    f"layer{layer_idx}.kv_einsum",
                )
                if not success:
                    print(f"Failed to set kv_einsum for layer {layer_idx}")
        else:
            # MHA: combined qkv_einsum
            # qkv_einsum.w: [3, num_heads, features, head_dim]
            if "q" in qkv_weights and "k" in qkv_weights and "v" in qkv_weights:
                q_weight = qkv_weights["q"]  # [embed_dim, num_heads * head_dim]
                k_weight = qkv_weights["k"]
                v_weight = qkv_weights["v"]
                # Reshape each to [embed_dim, num_heads, head_dim]
                q_weight = q_weight.reshape(
                    gemma_config.embed_dim,
                    gemma_config.num_heads,
                    gemma_config.head_dim,
                )
                k_weight = k_weight.reshape(
                    gemma_config.embed_dim,
                    gemma_config.num_heads,
                    gemma_config.head_dim,
                )
                v_weight = v_weight.reshape(
                    gemma_config.embed_dim,
                    gemma_config.num_heads,
                    gemma_config.head_dim,
                )
                # Stack to [3, embed_dim, num_heads, head_dim]
                qkv_weight = jnp.stack([q_weight, k_weight, v_weight], axis=0)
                # Transpose to [3, num_heads, embed_dim, head_dim]
                qkv_weight = jnp.transpose(qkv_weight, (0, 2, 1, 3))
                set_nested(model, (*nnx_layer, "qkv_einsum", "w"), qkv_weight)

    # Show keys in state_dict that are NOT in weight_mapping (potential missing mappings)
    unmapped_keys = [
        k
        for k in state_dict.keys()
        if k not in weight_mapping
        and "q_proj" not in k
        and "k_proj" not in k
        and "v_proj" not in k
    ]
    if unmapped_keys:
        # Only print if significantly many unmapped keys remain that aren't vision/multimodal related
        pass

    for hf_name, nnx_path in weight_mapping.items():
        if hf_name in state_dict:
            try:
                weight = np_to_jax(state_dict[hf_name])

                # Handle output projection (attn_vec_einsum)
                # attn_vec_einsum.w: [num_heads, head_dim, features]
                # HuggingFace o_proj: [embed_dim, num_heads * head_dim]
                if "o_proj.weight" in hf_name:
                    weight = weight.T  # [embed_dim, num_heads * head_dim] -> [num_heads * head_dim, embed_dim]
                    weight = weight.reshape(
                        gemma_config.num_heads,
                        gemma_config.head_dim,
                        gemma_config.embed_dim,
                    )
                    success = set_nested(model, nnx_path, weight, hf_name)
                    if success:
                        loaded_count += 1
                    continue

                # Transpose linear weights (HuggingFace: [out, in] -> NNX: [in, out])
                # But NOT embeddings - they are [vocab, embed] in both formats
                is_embedding = "embed_tokens" in hf_name
                if "weight" in hf_name and weight.ndim == 2 and not is_embedding:
                    weight = weight.T

                success = set_nested(model, nnx_path, weight, hf_name)
                if success:
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

"""
Converter for loading HuggingFace GPT-2 pretrained weights into NNX models.

Uses safetensors and huggingface_hub to load weights directly without PyTorch or Transformers dependencies.
"""

from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

from ponderttt.models.gpt2_nnx import GPT2LMHeadModel, GPT2Config


def load_huggingface_weights(
    model: GPT2LMHeadModel,
    model_name: str = "gpt2",
    cache_dir: Optional[Path] = None
) -> GPT2LMHeadModel:
    """Load pretrained HuggingFace weights into NNX model using safetensors.

    Args:
        model: NNX GPT-2 model to load weights into
        model_name: HuggingFace model name (gpt2, gpt2-medium, etc.)
        cache_dir: Optional cache directory for downloaded weights

    Returns:
        Model with loaded weights
    """
    print(f"Downloading/Loading weights for {model_name} (safetensors)...")
    
    try:
        # Try loading model.safetensors (standard for modern HF repos)
        file_path = hf_hub_download(
            repo_id=model_name,
            filename="model.safetensors",
            cache_dir=cache_dir
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not find 'model.safetensors' for {model_name}. "
            "Please ensure the model repo contains safetensors weights.\n"
            f"Error: {str(e)}"
        )

    print(f"Loading weights from {file_path}...")
    state_dict = load_file(file_path)

    print("Converting weights to JAX format...")
    _load_weights_from_state_dict(model, state_dict)

    print(f"OK Successfully loaded {model_name} weights")
    return model


def _load_weights_from_state_dict(model: GPT2LMHeadModel, state_dict: dict[str, Any]) -> None:
    """Load weights from safetensors numpy dictionary into NNX model.

    Args:
        model: NNX model to load into
        state_dict: Dictionary of numpy arrays from safetensors
    """
    config = model.config

    def _set_param_value(param: nnx.Param | None, value: np.ndarray | jnp.ndarray, name: str) -> None:
        if param is None:
            raise ValueError(f"Parameter '{name}' is not initialized")
        # Convert numpy to jax array if needed
        if isinstance(value, np.ndarray):
            value = jnp.array(value)
        # NNX Variable in-place update (reading uses [...], writing uses .value)
        param.value = value

    def _pad_to_match(param: nnx.Param | None, value: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        if isinstance(value, np.ndarray):
            value = jnp.array(value)

        if param is None:
            return value
        if value.shape == param[...].shape:
            return value
        # Only pad vocab dimension if needed (assumes vocab dim is first)
        if value.shape[0] < param[...].shape[0]:
            pad_rows = param[...].shape[0] - value.shape[0]
            pad_config = [(0, pad_rows)] + [(0, 0) for _ in range(value.ndim - 1)]
            return jnp.pad(value, pad_config)
        return value
    
    def get_weight(name: str) -> np.ndarray:
        """Helper to find weight with or without 'transformer.' prefix."""
        # Try with "transformer." prefix first (standard HF GPT2)
        if f"transformer.{name}" in state_dict:
            return state_dict[f"transformer.{name}"]
        # Try without prefix
        if name in state_dict:
            return state_dict[name]
        # Fallback for blocks: sometimes h.0 is directly at root
        if name.startswith("h.") and name in state_dict:
             return state_dict[name]
        
        raise KeyError(f"Could not find weight for {name}")

    # Token embeddings
    _set_param_value(
        model.transformer.wte.embedding,
        _pad_to_match(model.transformer.wte.embedding, get_weight("wte.weight")),
        "transformer.wte.embedding",
    )

    # Position embeddings
    _set_param_value(
        model.transformer.wpe.embedding,
        get_weight("wpe.weight"),
        "transformer.wpe.embedding",
    )

    # Transformer blocks
    for i in range(config.n_layer):
        block = model.transformer.h[i]
        
        # Layer norm 1
        _set_param_value(
            block.ln_1.scale,
            get_weight(f"h.{i}.ln_1.weight"),
            f"h.{i}.ln_1.scale",
        )
        _set_param_value(
            block.ln_1.bias,
            get_weight(f"h.{i}.ln_1.bias"),
            f"h.{i}.ln_1.bias",
        )

        # Attention: combined QKV
        # GPT-2 Conv1D weights are [in, out]
        _set_param_value(
            block.attn.c_attn.kernel,
            get_weight(f"h.{i}.attn.c_attn.weight"),
            f"h.{i}.attn.c_attn.kernel",
        )
        _set_param_value(
            block.attn.c_attn.bias,
            get_weight(f"h.{i}.attn.c_attn.bias"),
            f"h.{i}.attn.c_attn.bias",
        )

        # Attention: output projection
        _set_param_value(
            block.attn.c_proj.kernel,
            get_weight(f"h.{i}.attn.c_proj.weight"),
            f"h.{i}.attn.c_proj.kernel",
        )
        _set_param_value(
            block.attn.c_proj.bias,
            get_weight(f"h.{i}.attn.c_proj.bias"),
            f"h.{i}.attn.c_proj.bias",
        )

        # Layer norm 2
        _set_param_value(
            block.ln_2.scale,
            get_weight(f"h.{i}.ln_2.weight"),
            f"h.{i}.ln_2.scale",
        )
        _set_param_value(
            block.ln_2.bias,
            get_weight(f"h.{i}.ln_2.bias"),
            f"h.{i}.ln_2.bias",
        )

        # MLP: first projection
        _set_param_value(
            block.mlp.c_fc.kernel,
            get_weight(f"h.{i}.mlp.c_fc.weight"),
            f"h.{i}.mlp.c_fc.kernel",
        )
        _set_param_value(
            block.mlp.c_fc.bias,
            get_weight(f"h.{i}.mlp.c_fc.bias"),
            f"h.{i}.mlp.c_fc.bias",
        )

        # MLP: second projection
        _set_param_value(
            block.mlp.c_proj.kernel,
            get_weight(f"h.{i}.mlp.c_proj.weight"),
            f"h.{i}.mlp.c_proj.kernel",
        )
        _set_param_value(
            block.mlp.c_proj.bias,
            get_weight(f"h.{i}.mlp.c_proj.bias"),
            f"h.{i}.mlp.c_proj.bias",
        )

    # Final layer norm
    _set_param_value(
        model.transformer.ln_f.scale,
        get_weight("ln_f.weight"),
        "transformer.ln_f.scale",
    )
    _set_param_value(
        model.transformer.ln_f.bias,
        get_weight("ln_f.bias"),
        "transformer.ln_f.bias",
    )

    # LM head (only if not using weight tying)
    if not model.tie_word_embeddings:
        try:
            # Try standard key
            lm_head_weight = get_weight("lm_head.weight")
            _set_param_value(
                model.lm_head.kernel,
                _pad_to_match(model.lm_head.kernel, lm_head_weight).T,
                "lm_head.kernel",
            )
        except KeyError:
            # Often weights are tied, so explicit lm_head weights might be missing.
            print("Warning: 'lm_head.weight' not found in checkpoint. "
                  "Assuming weights are tied or handled elsewhere.")


def save_checkpoint(
    model: GPT2LMHeadModel,
    checkpoint_path: Path,
    config: Optional[GPT2Config] = None,
) -> None:
    """Save NNX model checkpoint using Orbax.

    Args:
        model: NNX model to save
        checkpoint_path: Path to save checkpoint
        config: Optional config to save alongside weights
    """
    try:
        import orbax.checkpoint as ocp
    except ImportError:
        raise ImportError("orbax is required for checkpointing. Install with: pip install orbax-checkpoint")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Create checkpoint manager
    checkpointer = ocp.PyTreeCheckpointer()

    # Extract model state
    graphdef, state = nnx.split(model)

    # Save state
    print(f"Saving checkpoint to {checkpoint_path}...")
    checkpointer.save(checkpoint_path / "model", state)

    # Save config if provided
    if config is not None:
        import json

        config_dict = {
            "vocab_size": config.vocab_size,
            "n_positions": config.n_positions,
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "dropout": config.dropout,
            "layer_norm_epsilon": config.layer_norm_epsilon,
            "tie_word_embeddings": model.tie_word_embeddings,
        }
        with open(checkpoint_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    print(f"OK Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    seed: int = 0,
) -> tuple[GPT2LMHeadModel, GPT2Config]:
    """Load NNX model checkpoint using Orbax.

    Args:
        checkpoint_path: Path to checkpoint directory
        seed: Random seed for model initialization

    Returns:
        (model, config) tuple
    """
    try:
        import orbax.checkpoint as ocp
    except ImportError:
        raise ImportError("orbax is required for checkpointing. Install with: pip install orbax-checkpoint")

    checkpoint_path = Path(checkpoint_path)

    # Load config
    import json
    with open(checkpoint_path / "config.json") as f:
        config_dict = json.load(f)
    tie_word_embeddings = config_dict.pop("tie_word_embeddings", True)
    config = GPT2Config(**config_dict)

    # Create model
    rngs = nnx.Rngs(seed)
    model = GPT2LMHeadModel(config, rngs, tie_word_embeddings=tie_word_embeddings)

    # Get model graphdef and state
    graphdef, state = nnx.split(model)

    # Load checkpoint
    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(checkpoint_path / "model")

    # Merge back into model
    model = nnx.merge(graphdef, restored_state)

    print(f"OK Checkpoint loaded from {checkpoint_path}")
    return model, config
"""
Converter for loading HuggingFace GPT-2 pretrained weights into NNX models.

Converts PyTorch checkpoints from HuggingFace to Flax NNX format.
"""

from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
from flax import nnx

from ponderttt.models.gpt2_nnx import GPT2LMHeadModel, GPT2Config


def load_huggingface_weights(
    model: GPT2LMHeadModel,
    model_name: str = "gpt2",
    cache_dir: Optional[Path] = None
) -> GPT2LMHeadModel:
    """Load pretrained HuggingFace weights into NNX model.

    Args:
        model: NNX GPT-2 model to load weights into
        model_name: HuggingFace model name (gpt2, gpt2-medium, etc.)
        cache_dir: Optional cache directory for downloaded weights

    Returns:
        Model with loaded weights

    Note:
        This function temporarily imports transformers to load weights,
        but transformers is not required at runtime after weights are loaded.
    """
    try:
        from transformers import GPT2LMHeadModel as HFModel
    except ImportError:
        raise ImportError(
            "transformers is required for loading pretrained weights. "
            "Install with: pip install transformers"
        )

    # Load HuggingFace model (PyTorch)
    print(f"Loading pretrained weights from {model_name}...")
    hf_model = HFModel.from_pretrained(model_name, cache_dir=cache_dir)
    hf_state_dict = hf_model.state_dict()

    # Convert PyTorch tensors to JAX arrays
    print("Converting weights to JAX format...")
    _load_weights_from_state_dict(model, hf_state_dict)

    print(f"OK Successfully loaded {model_name} weights")
    return model


def _load_weights_from_state_dict(model: GPT2LMHeadModel, state_dict: dict[str, Any]) -> None:
    """Load weights from PyTorch state dict into NNX model.

    Args:
        model: NNX model to load into
        state_dict: PyTorch state dict from HuggingFace
    """
    config = model.config

    def _set_param_value(param: nnx.Param | None, value: jnp.ndarray, name: str) -> None:
        if param is None:
            raise ValueError(f"Parameter '{name}' is not initialized")
        param.value = value

    # Token embeddings
    _set_param_value(
        model.transformer.wte.embedding,
        jnp.array(state_dict["transformer.wte.weight"].numpy()),
        "transformer.wte.embedding",
    )

    # Position embeddings
    _set_param_value(
        model.transformer.wpe.embedding,
        jnp.array(state_dict["transformer.wpe.weight"].numpy()),
        "transformer.wpe.embedding",
    )

    # Transformer blocks
    for i in range(config.n_layer):
        block = model.transformer.h[i]
        prefix = f"transformer.h.{i}"

        # Layer norm 1
        _set_param_value(
            block.ln_1.scale,
            jnp.array(state_dict[f"{prefix}.ln_1.weight"].numpy()),
            f"{prefix}.ln_1.scale",
        )
        _set_param_value(
            block.ln_1.bias,
            jnp.array(state_dict[f"{prefix}.ln_1.bias"].numpy()),
            f"{prefix}.ln_1.bias",
        )

        # Attention: combined QKV
        # Note: GPT-2's Conv1D already uses [in_features, out_features] like Flax
        qkv_weight = state_dict[f"{prefix}.attn.c_attn.weight"].numpy()
        qkv_bias = state_dict[f"{prefix}.attn.c_attn.bias"].numpy()
        _set_param_value(
            block.attn.c_attn.kernel,
            jnp.array(qkv_weight),
            f"{prefix}.attn.c_attn.kernel",
        )
        _set_param_value(
            block.attn.c_attn.bias,
            jnp.array(qkv_bias),
            f"{prefix}.attn.c_attn.bias",
        )

        # Attention: output projection
        proj_weight = state_dict[f"{prefix}.attn.c_proj.weight"].numpy()
        proj_bias = state_dict[f"{prefix}.attn.c_proj.bias"].numpy()
        _set_param_value(
            block.attn.c_proj.kernel,
            jnp.array(proj_weight),
            f"{prefix}.attn.c_proj.kernel",
        )
        _set_param_value(
            block.attn.c_proj.bias,
            jnp.array(proj_bias),
            f"{prefix}.attn.c_proj.bias",
        )

        # Layer norm 2
        _set_param_value(
            block.ln_2.scale,
            jnp.array(state_dict[f"{prefix}.ln_2.weight"].numpy()),
            f"{prefix}.ln_2.scale",
        )
        _set_param_value(
            block.ln_2.bias,
            jnp.array(state_dict[f"{prefix}.ln_2.bias"].numpy()),
            f"{prefix}.ln_2.bias",
        )

        # MLP: first projection
        # Note: GPT-2's Conv1D already uses [in_features, out_features] like Flax
        fc_weight = state_dict[f"{prefix}.mlp.c_fc.weight"].numpy()
        fc_bias = state_dict[f"{prefix}.mlp.c_fc.bias"].numpy()
        _set_param_value(
            block.mlp.c_fc.kernel,
            jnp.array(fc_weight),
            f"{prefix}.mlp.c_fc.kernel",
        )
        _set_param_value(
            block.mlp.c_fc.bias,
            jnp.array(fc_bias),
            f"{prefix}.mlp.c_fc.bias",
        )

        # MLP: second projection
        proj_weight = state_dict[f"{prefix}.mlp.c_proj.weight"].numpy()
        proj_bias = state_dict[f"{prefix}.mlp.c_proj.bias"].numpy()
        _set_param_value(
            block.mlp.c_proj.kernel,
            jnp.array(proj_weight),
            f"{prefix}.mlp.c_proj.kernel",
        )
        _set_param_value(
            block.mlp.c_proj.bias,
            jnp.array(proj_bias),
            f"{prefix}.mlp.c_proj.bias",
        )

    # Final layer norm
    _set_param_value(
        model.transformer.ln_f.scale,
        jnp.array(state_dict["transformer.ln_f.weight"].numpy()),
        "transformer.ln_f.scale",
    )
    _set_param_value(
        model.transformer.ln_f.bias,
        jnp.array(state_dict["transformer.ln_f.bias"].numpy()),
        "transformer.ln_f.bias",
    )

    # LM head (only if not using weight tying)
    if not model.tie_word_embeddings and "lm_head.weight" in state_dict:
        lm_head_weight = state_dict["lm_head.weight"].numpy()
        _set_param_value(
            model.lm_head.kernel,
            jnp.array(lm_head_weight).T,
            "lm_head.kernel",
        )


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


def convert_and_save_pretrained(
    model_name: str = "gpt2",
    output_dir: Path = Path("checkpoints"),
    tie_word_embeddings: bool = True
) -> None:
    """Convert HuggingFace checkpoint to NNX format and save.

    This is a one-time conversion utility. After conversion, transformers
    is no longer needed.

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save converted checkpoint
        tie_word_embeddings: Whether to use weight tying
    """
    from ponderttt.models.gpt2_nnx import load_gpt2_model

    # Create NNX model
    print(f"Creating NNX model for {model_name}...")
    model, config = load_gpt2_model(model_name, tie_word_embeddings=tie_word_embeddings)

    # Load pretrained weights from HuggingFace
    model = load_huggingface_weights(model, model_name)

    # Save in NNX format
    checkpoint_path = output_dir / model_name
    save_checkpoint(model, checkpoint_path, config)

    print(f"\nOK Converted {model_name} to NNX format")
    print(f"  Saved to: {checkpoint_path}")
    print("  You can now use this checkpoint without transformers dependency")


if __name__ == "__main__":
    """Example: Convert all GPT-2 variants to NNX format."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert HuggingFace GPT-2 to NNX format")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Model variant to convert"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Output directory for converted checkpoints"
    )
    parser.add_argument(
        "--no-tie-embeddings",
        action="store_true",
        help="Don't tie word embeddings (uses more parameters)"
    )

    args = parser.parse_args()

    convert_and_save_pretrained(
        model_name=args.model,
        output_dir=args.output_dir,
        tie_word_embeddings=not args.no_tie_embeddings
    )

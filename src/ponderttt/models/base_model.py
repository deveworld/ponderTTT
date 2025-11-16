"""
Base transformer language model.
"""

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM


@dataclass
class ModelConfig:
    """Configuration for base model."""
    model_name: str = "gpt2"
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False
    mesh: jax.sharding.Mesh | None = None
    shard_params: bool = False  # Enable parameter sharding for TPU Pods


def apply_sharding_to_params(params, mesh: jax.sharding.Mesh):
    """
    Apply FSDP (Fully Sharded Data Parallel) sharding to model parameters.

    This function implements FSDP for memory-efficient training on TPU Pods
    by sharding parameters across devices. Unlike pure data parallelism where
    parameters are replicated, FSDP shards them to reduce per-device memory usage.

    Sharding strategy (FSDP):
    - Embeddings: Shard along vocab dimension (first axis) using first mesh axis
    - Large kernels (>= 2D): Shard along output dimension (second axis) using first mesh axis
    - Biases and LayerNorm: Replicate (no sharding)
    - Small parameters: Replicate (no sharding)

    For a mesh with shape (N, 1) and axes ('batch', 'model'):
    - The 'batch' axis (size N) is used for FSDP sharding
    - Parameters are sharded N-way across devices
    - During forward pass, parameters are gathered via AllGather
    - During backward pass, gradients are reduced via Reduce-Scatter

    Args:
        params: Parameter tree from model
        mesh: JAX mesh for distributed computation

    Returns:
        Sharded parameters with sharding constraints applied

    Note:
        Safe to call multiple times - will not re-shard already sharded parameters.
    """
    from jax.lax import with_sharding_constraint

    def get_sharding_for_param(name, param):
        """Determine sharding spec based on parameter name and shape."""
        # Parameters smaller than this threshold are always replicated
        MIN_SIZE_FOR_SHARDING = 1024

        param_size = param.size

        # Small parameters: always replicate
        if param_size < MIN_SIZE_FOR_SHARDING:
            return NamedSharding(mesh, P())

        # Large embeddings: shard along vocabulary dimension (axis 0)
        if 'embed' in name.lower() and param.ndim >= 2:
            # vocab_size × embed_dim: shard vocab dimension
            return NamedSharding(mesh, P('batch', None))

        # Large weight matrices: shard along output dimension (axis 1)
        elif ('kernel' in name.lower() or 'weight' in name.lower()) and param.ndim >= 2:
            # input_dim × output_dim: shard output dimension
            return NamedSharding(mesh, P(None, 'batch'))

        # Biases, LayerNorm scales, and other 1D parameters: replicate
        elif 'bias' in name.lower() or 'scale' in name.lower() or param.ndim < 2:
            return NamedSharding(mesh, P())

        # Default: replicate for safety
        else:
            return NamedSharding(mesh, P())

    def shard_tree(path, subtree):
        """Recursively apply sharding to parameter tree."""
        if isinstance(subtree, dict):
            return {k: shard_tree(path + [k], v) for k, v in subtree.items()}
        elif isinstance(subtree, jnp.ndarray):
            param_name = '.'.join(path)
            sharding = get_sharding_for_param(param_name, subtree)

            # Use device_put for placement
            sharded_param = jax.device_put(subtree, sharding)

            # Apply sharding constraint
            sharded_param = with_sharding_constraint(sharded_param, sharding)

            return sharded_param
        else:
            return subtree

    return shard_tree([], params)


class TransformerLM(nn.Module):
    """
    Transformer language model wrapper.

    Uses HuggingFace Flax models as backbone.

    Attributes:
        config: Model configuration
    """
    config: ModelConfig

    def setup(self):
        """Initialize the model."""
        # Load pretrained model
        self.model = FlaxAutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=self.config.dtype,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict:
        """
        Forward pass through transformer.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            deterministic: Whether to use dropout
            output_hidden_states: Return hidden states
            output_attentions: Return attention weights

        Returns:
            Dictionary with:
                - logits: Output logits [batch, seq_len, vocab_size]
                - hidden_states: List of hidden states (if requested)
                - attentions: List of attention weights (if requested)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            train=not deterministic,
        )

        result = {
            'logits': outputs.logits,
        }

        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states

        if output_attentions:
            result['attentions'] = outputs.attentions

        return result

    def compute_loss(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict]:
        """
        Compute language modeling loss.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask

        Returns:
            loss: Cross-entropy loss
            metrics: Dictionary with auxiliary metrics
        """
        # Labels are input_ids shifted left
        labels = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]

        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1]

        # Forward pass
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=True,
        )

        logits = outputs['logits']

        # Compute cross-entropy loss
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # One-hot encode labels
        labels_one_hot = jax.nn.one_hot(labels_flat, vocab_size)

        # Cross-entropy
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        loss = -jnp.sum(labels_one_hot * log_probs, axis=-1)

        # Mask padding tokens
        if attention_mask is not None:
            mask_flat = attention_mask[:, 1:].reshape(-1)
            loss = loss * mask_flat
            loss = jnp.sum(loss) / jnp.sum(mask_flat)
        else:
            loss = jnp.mean(loss)

        # Compute perplexity
        perplexity = jnp.exp(loss)

        metrics = {
            'loss': loss,
            'perplexity': perplexity,
        }

        return loss, metrics


def load_model(
    model_name: str = "gpt2",
    dtype: jnp.dtype = jnp.float32,
    mesh: jax.sharding.Mesh | None = None,
    shard_params: bool = False,
) -> tuple[TransformerLM, AutoTokenizer]:
    """
    Load pretrained model and tokenizer.

    Args:
        model_name: HuggingFace model name
        dtype: Data type for model
        mesh: JAX mesh for distributed computation (TPU Pods)
        shard_params: Whether to shard parameters across devices

    Returns:
        model: TransformerLM instance
        tokenizer: Tokenizer
    """
    config = ModelConfig(
        model_name=model_name,
        dtype=dtype,
        mesh=mesh,
        shard_params=shard_params,
    )

    # Initialize model
    model = TransformerLM(config=config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def initialize_sharded_model(
    model: TransformerLM,
    rng: jax.random.PRNGKey,
    input_shape: tuple[int, int],
) -> dict:
    """
    Initialize model parameters with optional sharding for TPU Pods.

    Args:
        model: TransformerLM instance
        rng: Random number generator key
        input_shape: Input shape (batch_size, seq_length)

    Returns:
        params: Initialized (and potentially sharded) parameters
    """
    # Create dummy input for initialization
    dummy_input = jnp.ones(input_shape, dtype=jnp.int32)

    # Initialize parameters
    variables = model.init(rng, dummy_input)
    params = variables['params']

    # Apply sharding if configured
    if model.config.shard_params and model.config.mesh is not None:
        params = apply_sharding_to_params(params, model.config.mesh)

    return params


def count_parameters(params) -> int:
    """
    Count number of parameters in a parameter tree.

    Args:
        params: JAX parameter tree

    Returns:
        Total number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def inspect_sharding(params, max_params: int = 20) -> None:
    """
    Print sharding information for model parameters.

    Args:
        params: Parameter tree
        max_params: Maximum number of parameters to display

    Example output:
        transformer.wte.embedding: (50257, 768) -> P('batch', None)
        transformer.h.0.attn.c_attn.kernel: (768, 2304) -> P(None, 'batch')
        transformer.h.0.attn.c_attn.bias: (2304,) -> P() [replicated]
    """
    if jax.process_index() != 0:
        return  # Only print from main process

    print("\n" + "=" * 80)
    print("Parameter Sharding Inspection")
    print("=" * 80)

    def inspect_tree(path, subtree, count=None):
        """Recursively inspect parameter tree."""
        if count is None:
            count = [0]
        if isinstance(subtree, dict):
            for k, v in subtree.items():
                inspect_tree(path + [k], v, count)
        elif isinstance(subtree, jnp.ndarray):
            if count[0] < max_params:
                param_name = '.'.join(path)
                shape = subtree.shape
                sharding = subtree.sharding if hasattr(subtree, 'sharding') else "unknown"

                # Extract PartitionSpec if available
                if hasattr(sharding, 'spec'):
                    spec = sharding.spec
                    is_replicated = all(s is None for s in spec)
                    spec_str = f"P{spec}" if not is_replicated else "P() [replicated]"
                else:
                    spec_str = str(sharding)

                print(f"  {param_name:60s} {str(shape):20s} -> {spec_str}")
                count[0] += 1
            elif count[0] == max_params:
                print(f"  ... ({sum(1 for _ in jax.tree_util.tree_leaves(params)) - max_params} more parameters)")
                count[0] += 1

    inspect_tree([], params)
    total_params = count_parameters(params)
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print("=" * 80 + "\n")

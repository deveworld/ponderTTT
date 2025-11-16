"""
Example: Using PonderTTT with TPU Pod sharding support.

This example demonstrates how to initialize and use PonderTTT models
on Google Cloud TPU Pods (v4-64) with proper parameter sharding.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from ponderttt.models import (
    load_model,
    initialize_sharded_model,
)
from ponderttt.utils import initialize_jax_distributed, create_mesh
from transformers import PreTrainedTokenizer

# Step 1: Initialize JAX for multi-host distributed training
initialize_jax_distributed()

print(f"JAX process count: {jax.process_count()}")
print(f"JAX process index: {jax.process_index()}")
print(f"JAX local devices: {jax.local_device_count()}")

# Step 2: Create device mesh for TPU Pod
# For TPU v4-64: 8 hosts × 8 chips = 64 devices
# Mesh shape: (batch=8, model=8) for data + model parallelism
# OR: (batch=64, model=1) for pure data parallelism
mesh = create_mesh(mesh_shape=(8, 8), axis_names=('batch', 'model'))

print(f"Created mesh: {mesh}")
print(f"Mesh shape: {mesh.shape}")
print(f"Mesh axis names: {mesh.axis_names}")

# Step 3: Load model with sharding support
model, tokenizer_raw = load_model(
    model_name="gpt2",  # or "codegen-350M-multi", "gpt2-medium", etc.
    dtype=jnp.float32,
    mesh=mesh,
    shard_params=True,  # Enable parameter sharding
)
tokenizer = cast(PreTrainedTokenizer, tokenizer_raw)

print(f" Model loaded: {model.config.model_name}")

# Step 4: Initialize model parameters with sharding
rng = jax.random.PRNGKey(42)
params = initialize_sharded_model(
    model=model,
    rng=rng,
    input_shape=(2, 1024),  # (batch_size, seq_length)
)

print(" Parameters initialized and sharded")
print(f"Parameter structure: {jax.tree_util.tree_map(lambda x: x.shape, params)}")

# Step 5: Example forward pass
input_text = "def fibonacci(n):"
input_ids = tokenizer.encode(input_text, return_tensors="jax")

# Shard input data across devices
input_sharding = NamedSharding(mesh, P('batch', None))
sharded_input = jax.device_put(input_ids, input_sharding)

# Forward pass
outputs = cast(dict[str, Any], model.apply({'params': params}, sharded_input))
logits = outputs['logits']

print(" Forward pass complete")
print(f"Input shape: {sharded_input.shape}")
print(f"Output logits shape: {logits.shape}")

# Step 6: Verify sharding
print("\nSharding verification:")
print(f"Input sharding: {sharded_input.sharding}")
print(f"Output sharding: {logits.sharding}")

# Check parameter sharding
param_sample = jax.tree_util.tree_leaves(params)[0]
print(f"Parameter sharding example: {param_sample.sharding}")

print("\n TPU Pod sharding example complete!")

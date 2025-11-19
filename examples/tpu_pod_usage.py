"""
Example: Using PonderTTT with TPU Pod sharding support.

This example demonstrates how to initialize and use PonderTTT models
on Google Cloud TPU Pods (v4-64) with proper parameter sharding.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from ponderttt.data import get_tokenizer
from ponderttt.models import load_ttt_model
from ponderttt.utils import create_mesh, initialize_jax_distributed

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

# Step 3: Load model (NNX) and tokenizer
model, config = load_ttt_model(
    model_name="gpt2",
    seed=42,
    load_pretrained=False,
)
model.train()
tokenizer = get_tokenizer("gpt2")

print(f" Model loaded: {config.n_layer} layers, hidden {config.n_embd}")

# Step 4: Prepare input and shard across mesh
input_text = "def fibonacci(n):\n    if n <= 1:\n        return n"
encoded = tokenizer.encode(input_text)
token_ids = encoded.ids[: config.n_positions]
if len(token_ids) < config.n_positions:
    token_ids += [tokenizer.token_to_id("<|pad|>")] * (config.n_positions - len(token_ids))
input_ids = jnp.array([token_ids], dtype=jnp.int32)

input_sharding = NamedSharding(mesh, P("batch", None))
sharded_input = jax.device_put(input_ids[:, :512], input_sharding)

# Forward pass (TTT disabled for inference demo)
outputs = cast(dict[str, Any], model(sharded_input, use_ttt=False))
logits = outputs["logits"]

print(" Forward pass complete")
print(f"Input shape: {sharded_input.shape}")
print(f"Output logits shape: {logits.shape}")

# Step 6: Verify sharding
print("\nSharding verification:")
print(f"Input sharding: {sharded_input.sharding}")
print(f"Output sharding: {logits.sharding}")

print("\n TPU Pod sharding example complete!")

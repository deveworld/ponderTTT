# Copyright 2024 The Flax Authors.
# Licensed under the Apache License, Version 2.0

"""Stub for sow_lib - intermediate activation sowing (not needed for PonderTTT)."""

import dataclasses
from flax import nnx
import jax


@dataclasses.dataclass(frozen=True)
class SowConfig:
    """Stub SowConfig - all sowing disabled."""

    embeddings: bool = False
    rs_after_attention: bool = False
    rs_after_ffw: bool = False
    mlp_hidden_topk: int = 0
    attn_logits_topk: int = 0

    def maybe_sow_embeddings(self, embeddings: jax.Array, module: nnx.Module):
        pass

    def maybe_sow_rs_after_attention(self, activations: jax.Array, module: nnx.Module):
        pass

    def maybe_sow_rs_after_ffw(self, activations: jax.Array, module: nnx.Module):
        pass

    def maybe_sow_mlp_hidden_topk(self, activations: jax.Array, module: nnx.Module):
        pass

    def maybe_sow_attn_logits_topk(self, logits: jax.Array, module: nnx.Module):
        pass

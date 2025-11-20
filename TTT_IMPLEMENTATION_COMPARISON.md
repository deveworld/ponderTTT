# TTT Implementation Notes

This document summarizes how the current NNX code base lines up with the official TTT-LM reference implementation.

## Base language model
- GPT-2 blocks follow the reference architecture (pre-norm transformers, tied token embeddings / LM head).
- Weight tying is implemented directly in `TTTTransformerLM` and validated by `scripts/test_weight_tying.py`.
- The frozen slow weights are enforced with `jax.lax.stop_gradient`, so only the fast-weight adapter receives gradients during TTT updates.

## Fast-weight layer
- The `TTTLayer` port includes the components from the reference implementation: causal convolutions for Q/K, rotary position embeddings (cached for full sequence length, no mini-batch wrapping), per-head layer norm, learnable per-head learning rates, and multiplicative gating via `wg`. GELU uses the GPT-2-style approximate form consistently across fast/slow paths.
- LoRA adapters provide low-rank Q/V adaptations with a lightweight self-attention readout; this is a simplified integration rather than modifying the base attention projections.
- UPDATE actions call `run_chunk_step` repeatedly, so UPDATE_1/2/4 literally execute 1/2/4 fast-weight updates on the current chunk.
- The layer reports the same diagnostic statistics (`ttt_loss_*`, `ssl_target_variance`) used for monitoring in TTT-LM.

## Training objective
- The trainers optimize the language-modeling cross-entropy plus a small SSL auxiliary loss (XV−XK MSE) emitted by the TTT layer (configurable weight).

## Remaining gaps
- Internal TTT updates are always reset between chunk sequences for stability. The reference implementation optionally keeps weights alive across longer contexts; we may re-introduce that once policy training requires it.
- LoRA integration is simplified (separate self-attention passthrough) rather than fully merged into the base attention projections.
- Safety: benchmark execution is gated behind `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS`; a hardened sandbox is still recommended for real runs.

These notes will be updated as additional parity work lands (e.g., auxiliary losses, extended horizons).

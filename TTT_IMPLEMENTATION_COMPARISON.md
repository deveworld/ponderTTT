# TTT Implementation Notes

This document summarizes how the current NNX code base lines up with the official TTT-LM reference implementation.

## Base language model
- GPT-2 blocks follow the reference architecture (pre-norm transformers, tied token embeddings / LM head).
- Weight tying is implemented directly in `TTTTransformerLM` and validated by `scripts/test_weight_tying.py`.
- The frozen slow weights are enforced with `jax.lax.stop_gradient`, so only the fast-weight adapter receives gradients during TTT updates.

## Fast-weight layer
- The `TTTLayer` port includes the components from the reference implementation: causal convolutions for Q/K, rotary position embeddings, per-head layer norm, learnable per-head learning rates, and multiplicative gating via `wg`.
- UPDATE actions call `run_chunk_step` repeatedly, so UPDATE_1/2/4 literally execute 1/2/4 fast-weight updates on the current chunk.
- The layer reports the same diagnostic statistics (`ttt_loss_*`, `ssl_target_variance`) used for monitoring in TTT-LM.

## Training objective
- The current trainers optimize the language-modeling cross-entropy only; the auxiliary self-supervised (XV−XK) loss from the original paper is not yet added to the objective. The stats are still exposed, so adding the extra term is straightforward in future work.

## Remaining gaps
- Internal TTT updates are always reset between chunk sequences for stability. The reference implementation optionally keeps weights alive across longer contexts; we may re-introduce that once policy training requires it.
- Safety-hardened execution for benchmark code is left to downstream tooling; the evaluation module executes completions directly in a sandboxed namespace but does not isolate system calls.

These notes will be updated as additional parity work lands (e.g., auxiliary losses, extended horizons).

# Contributing

Thanks for helping improve PonderTTT!

## Getting started
1. Install dependencies (see `README.md`). Use `uv` with editable installs.
2. Run the sanity checks:
   ```bash
   python scripts/quick_test.py
   python scripts/test_pipeline.py
   pytest tests/test_checkpointing.py  # optional but recommended
   ```
3. Use `uv run ...` for CLI commands so the correct environment is used. Tokenizers download from Hugging Face; set `HF_HOME` if desired.

## Development tips
- The code base is pure JAX/Flax NNX. Avoid reintroducing old Linen modules.
- Chunk-level semantics are centralized in `ponderttt/experiments/training_utils.py`. If you add new trainers, use that helper to keep SKIP/UPDATE_k behaviour consistent.
- New fast-weight variants should implement the same interface as `TTTLayer`/`LoRALayer`.
- Benchmarks execute user completions. Code execution is gated by `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS`; only enable it inside a sandbox.
- The streaming pipeline requires `seq_length % chunk_size == 0` and a tokenizer with `<|pad|>` present (enforced). If you add tokens, pass the resulting `vocab_size` to `load_ttt_model`.
- Policy training uses mini-batch PPO (`--ppo_minibatch_size`) with value clipping and PID anti-windup; keep new training loops consistent with that interface.

## Tests & formatting
- We rely on `pytest`, `ruff`, `pyright`, and `ty`. The CI expectation is:
  ```bash
  make check
  ```
- Long-running TPU/GPU tests are not part of CI; keep lightweight regressions under `scripts/` and `tests/`.

## Submitting changes
1. Create a branch (`feature/chunk-visualizer`, etc.).
2. Update documentation when behaviour changes; we treat the README/PLAN/PROJECT_STATUS as code.
3. Include clear commit messages and explain how you validated the change.
4. Open a PR referencing the relevant issue or research task; attach logs for `scripts/quick_test.py` and any other relevant tests.

We appreciate issues describing bugs, missing docs, or ideas for baselines/ablations. Thanks for contributing!

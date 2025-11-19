# PonderTTT

Adaptive, chunk-level Test-Time Training (TTT) for code generation models built with JAX/Flax NNX.

## What this project provides
- **Chunk-aware fast weights** – the base GPT-2 model stays frozen while a TTT or LoRA fast-weight module updates per chunk. Actions `SKIP / UPDATE_1 / UPDATE_2 / UPDATE_4` now correspond to 0/1/2/4 gradient-style updates on the selected chunk.
- **Budget-constrained policy learning** – a PPO + PID Lagrangian agent decides which action to take for each chunk by observing 32-D features (loss deltas, confidence, difficulty history, and remaining budget).
- **Streaming data pipeline** – The Stack v2 (dedup) is streamed directly from Software Heritage, padded with a dedicated `<|pad|>` token, and reshaped into `(batch, num_chunks, chunk_size)` tensors with aligned masks.
- **Executable evaluation** – HumanEval, MBPP, and ClassEval now call user provided `generate_fn` and run the associated tests to report pass@k scores.
- **Operational tooling** – scripts cover quick sanity checks, end-to-end integration, distributed readiness, TPU validation, and policy/baseline training.

The code base is TPU-first but runs on CPU or GPU. Everything is dependency-managed through `uv`.

## Installation
```bash
# Install uv if you do not have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project in editable mode (CPU only)
uv pip install -e .

# Optional extras
uv pip install -e .[dev]      # linting / pytest
uv pip install -e .[viz]      # matplotlib + seaborn
```

If you want CUDA support, install a CUDA-enabled `jaxlib` before `uv pip install -e .`:
```bash
uv pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Tokenizer weights are downloaded from Hugging Face; set `HF_HOME` or mirror the files locally if you are offline.

## Quick validation
Run the lightweight smoke tests once after installation:
```bash
python scripts/quick_test.py          # tokenizer/model/feature sanity
python scripts/test_pipeline.py       # end-to-end chunk pipeline
python scripts/test_distributed.py    # multi-host primitives (CPU friendly)
python scripts/test_weight_tying.py   # weight tying regression
```
For TPU deployments run `python scripts/validate_tpu_setup.py --multi_host` on all workers before launching long jobs.

## Baseline training
`train_baseline.py` consumes streamed chunks and applies the specified action to every chunk:
```bash
uv run python -m ponderttt.experiments.train_baseline \
    --model_scale 125m \
    --action UPDATE_2 \
    --max_chunks 200 \
    --output_dir outputs/baselines
```
Outputs include average loss/perplexity and the true compute multiplier accumulated over processed chunks. `--fast_weight_type lora --lora_rank 128` switches the fast-weight adapter.

## Policy training
The policy loop collects chunk sequences, queries the policy for each chunk, applies the requested number of fast-weight updates, and computes rewards from loss deltas while enforcing a cost budget.
```bash
uv run python -m ponderttt.experiments.train_policy \
    --model_scale 125m \
    --rollout_length 64 \
    --budget_limit 4.0 \
    --num_iterations 50 \
    --output_dir outputs/policy
```
Results are saved as JSON (training history); visualize them with `python scripts/visualize_results.py --results_file outputs/policy/...json`.

## Evaluation
Use `ponderttt.evaluation.benchmarks` to run executable pass@k tests once a policy/baseline checkpoint is ready. Provide a `generate_fn(prompt) -> Iterable[str]` that yields up to `k` completions; the suite compiles and executes them in-process.

## Project status
- Pure NNX GPT-2 implementation with chunk-level TTT
- Streaming data + chunk masks with dedicated padding
- Policy training loop with real budgets, history-aware features, and fixed GAE
- Executable benchmark suite
- ⏳ Large-scale experiments (TPU v4-64) once hardware is available

See `PLAN.md` for the research roadmap and `PROJECT_STATUS.md` for the current milestone tracker.

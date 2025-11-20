# PonderTTT

Adaptive, chunk-level Test-Time Training (TTT) for code generation models built with JAX/Flax NNX.

## What this project provides
- **Chunk-aware fast weights** – GPT-2 slow weights stay frozen while a TTT or strictly low-rank LoRA adapter updates per chunk. Actions `SKIP / UPDATE_1 / UPDATE_2 / UPDATE_4` map to 0/1/2/4 optimizer steps on the fast weights.
- **Budget-constrained policy learning** – PPO with PID Lagrangian uses 32-D mask-aware features, cost-penalized rewards, advantage normalization, clipped value loss, multi-epoch updates, grad clipping, and KL logging to respect compute budgets.
- **Streaming data pipeline** – The Stack v2 (dedup) streamed from Software Heritage with a required `seq_length % chunk_size == 0`, dedicated `<|pad|>` token, aligned masks, and cache keys that include language/tokenizer.
- **Executable evaluation (gated)** – HumanEval/MBPP/ClassEval helpers are present; unsafe exec is disabled unless `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS=1` is set in a trusted sandbox.
- **Operational tooling** – quick sanity tests, end-to-end pipeline checks, distributed/TPU validation, checkpointing, visualization, and baseline/policy trainers.

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
python scripts/compare_gpt2_nnx.py    # compare NNX logits to transformers (requires HF weights)
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
Use `ponderttt.evaluation.benchmarks` to run pass@k tests with a `generate_fn(prompt) -> Iterable[str]`. For safety, code execution is disabled by default; set `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS=1` only inside a trusted sandbox before calling `evaluate_all`.

## Project status
- Pure NNX GPT-2 implementation with chunk-level TTT
- Streaming data + chunk masks with dedicated padding
- Policy training loop with real budgets, history-aware features, and fixed GAE
- Executable benchmark suite
- ⏳ Large-scale experiments (TPU v4-64) once hardware is available

See `PLAN.md` for the research roadmap and `PROJECT_STATUS.md` for the current milestone tracker.

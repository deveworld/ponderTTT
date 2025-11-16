# Quick Start Guide

Get started with PonderTTT in 5 minutes!

## Prerequisites

This project uses **uv** for dependency management. Install it first:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
# Install dependencies
make install

# Or manually
uv pip install -e .
```

**Important**: This installs CPU-only JAX by default. For GPU support, see README.md.

## Quick Validation

Run a quick validation with synthetic data to verify the pipeline:

```bash
# Test baseline with synthetic data (should complete in 1-2 minutes on CPU)
uv run python -m ponderttt.experiments.train_baseline \
    --model_scale 125m \
    --action SKIP \
    --num_chunks 10
```

Expected output:
```
Loading GPT-2 model...
Generating synthetic data (10 chunks)...
Running SKIP baseline...
Chunk 1/10: loss=11.045
Chunk 2/10: loss=11.032
...
Average loss: 11.024
Total cost: 10.0 (baseline)
```

**What This Validates**:
- JAX/Flax installation working
- GPT-2 model loading
- TTT layer functionality
- Cost calculation correct
- Training loop functional

**Expected Results** (synthetic data):
- SKIP baseline: ~11.0 loss
- UPDATE_1/2/4: ~10.9 loss (marginal improvement expected)
- No errors or crashes

**Important Limitations**:
- Uses **random token data** (no semantic structure)
- TTT improvement is marginal (~0.1) on synthetic data
- Real code data needed for meaningful results
- GPU required for production training (CPU is 10-100× slower)

## First Experiment: Baseline Comparison

Compare different baseline strategies with synthetic data:

```bash
# Test all baselines (SKIP, UPDATE_1, UPDATE_2, UPDATE_4)
for action in SKIP UPDATE_1 UPDATE_2 UPDATE_4; do
    uv run python -m ponderttt.experiments.train_baseline \
        --model_scale 125m \
        --action $action \
        --num_chunks 20
done
```

**Expected Results** (synthetic data):
- SKIP: ~11.0 loss, cost=20.0 (1× per chunk)
- UPDATE_1: ~10.9 loss, cost=60.0 (3× per chunk)
- UPDATE_2: ~10.9 loss, cost=100.0 (5× per chunk)
- UPDATE_4: ~10.9 loss, cost=240.0 (12× per chunk)

**Time**: ~5-10 minutes on CPU, <1 minute on GPU

**Note**: These results are on random tokens. Real code data will show more differentiation.

## Second Experiment: Adaptive Policy (Requires GPU)

Train an adaptive policy using RL:

```bash
# Requires GPU - will be very slow on CPU
uv run python -m ponderttt.experiments.train_policy \
    --model_scale 125m \
    --num_iterations 10 \
    --output_dir outputs/policy
```

**Status**: Not yet run (requires GPU and real data)

This will:
- Initialize policy network
- Collect rollouts with current policy
- Update policy using PID-Lagrangian PPO
- Run 10 iterations for testing
- Save results to `outputs/policy/`

Expected time: ~10-15 minutes on GPU (hours on CPU)

## Visualize Results

After training, visualize the results:

```bash
python scripts/visualize_results.py \
    --results_file outputs/policy/125m/results.msgpack \
    --output_dir figures
```

This creates:
- `training_history.png`: Training curves
- `action_distribution.png`: Policy action frequencies

## Full Training (Requires GPU + Real Data)

For a complete training run with real code data:

```bash
# 1. Ensure you have GPU access
# 2. Obtain access to The Stack dataset (gated on HuggingFace)

# Train baseline for comparison
uv run python -m ponderttt.experiments.train_baseline \
    --model_scale 125m \
    --action UPDATE_1 \
    --output_dir outputs/baselines \
    --use_real_data

# Train adaptive policy
uv run python -m ponderttt.experiments.train_policy \
    --model_scale 125m \
    --num_iterations 100 \
    --output_dir outputs/policy \
    --use_real_data
```

**Current Blockers**:
1. The Stack dataset access (gated, requires approval)
2. GPU resources for production training

## Next Steps

1. **Try different scales**: Use `--model_scale 350m` or `1b`
2. **Compare baselines**: Train with different fixed actions (SKIP, UPDATE_2, UPDATE_4)
3. **Analyze results**: Compare costs and quality across methods
4. **Run ablations**: Modify feature sets or policy architecture

## Common Issues

### Out of Memory

If you get OOM errors:
- Reduce batch size in `src/ponderttt/experiments/config.py`
- Use smaller model scale
- Reduce chunk size

### Slow Training

If training is slow:
- Check JAX devices: `python -c "import jax; print(jax.devices())"`
- Reduce `max_train_examples` for faster iterations
- Use fewer rollout chunks

### Dependencies

If packages are missing:
```bash
make install
```

## Help

For more information:
- See [README.md](README.md) for full documentation
- See [PLAN.md](PLAN.md) for research plan
- Open an issue on GitHub for questions

## Example Workflow

### Current Status (CPU Validation)
```bash
# 1. Install dependencies
make install

# 2. Validate pipeline with synthetic data
uv run python -m ponderttt.experiments.train_baseline \
    --model_scale 125m \
    --action SKIP \
    --num_chunks 10

# 3. Compare all baselines
for action in SKIP UPDATE_1 UPDATE_2 UPDATE_4; do
    uv run python -m ponderttt.experiments.train_baseline \
        --model_scale 125m \
        --action $action \
        --num_chunks 20
done

# 4. Check outputs
ls outputs/baselines/125m/
```

### Future Workflow (GPU + Real Data)
```bash
# 1. Obtain The Stack dataset access
# 2. Secure GPU resources

# 3. Run real experiments
uv run python -m ponderttt.experiments.train_baseline \
    --model_scale 125m \
    --action UPDATE_1 \
    --use_real_data

# 4. Train policy
uv run python -m ponderttt.experiments.train_policy \
    --model_scale 125m \
    --num_iterations 100 \
    --use_real_data

# 5. Visualize results
uv run python scripts/visualize_results.py \
    --results_file outputs/policy/125m/results.msgpack
```

Happy validating!

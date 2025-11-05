# PonderTTT Quick Start Guide

**Ready to run WikiText-2 experiments!** 🚀

---

## Prerequisites

```bash
# Ensure you're in the project directory
cd /home/world/ponderttt

# Dependencies should already be installed
# If not, run: uv sync
```

---

## Quick Test (1-2 minutes)

Verify everything works:

```bash
# Run sanity checks
uv run python experiments/test_setup.py
```

**Expected output**: "All tests passed! ✅"

---

## Run Experiments

### Option 1: Run Everything (Recommended)

Run all baselines + adaptive in one command:

```bash
# GPU (recommended, ~3-4 hours)
uv run python experiments/wikitext2_experiment.py \
    --mode all \
    --num_epochs 3 \
    --max_train_batches 500 \
    --max_eval_batches 100 \
    --device cuda

# CPU (slow, ~20+ hours, for testing only)
uv run python experiments/wikitext2_experiment.py \
    --mode all \
    --num_epochs 1 \
    --max_train_batches 50 \
    --max_eval_batches 20 \
    --device cpu
```

This will run:
- Fixed-1 baseline (1 TTT iteration)
- Fixed-2 baseline (2 TTT iterations)
- Fixed-4 baseline (4 TTT iterations)
- Adaptive TTT (entropy-based allocation)

### Option 2: Run Individual Experiments

**Baseline only**:
```bash
uv run python experiments/wikitext2_experiment.py \
    --mode baseline \
    --ttt_iterations 2 \
    --num_epochs 3 \
    --device cuda
```

**Adaptive only**:
```bash
uv run python experiments/wikitext2_experiment.py \
    --mode adaptive \
    --num_epochs 3 \
    --device cuda
```

---

## Analyze Results

After experiments complete:

```bash
# Generate all visualizations and tables
uv run python experiments/analyze_wikitext2.py
```

**Outputs** (saved to `experiments/figures/`):
- `pareto_curve_wikitext2.png` - FLOPs vs Perplexity tradeoff
- `allocation_distribution.png` - How iterations are allocated
- `training_curves.png` - Validation perplexity over epochs

**Console output**:
- Formatted results table
- Summary statistics

---

## Expected Results

Based on Phase 1 findings, we expect:

| Metric | Target | Expected |
|--------|--------|----------|
| **FLOPs Reduction** | ≥20% | 30-40% |
| **Quality Loss** | <5% | <1% |
| **Allocation Accuracy** | >85% | 90-95% |

**Baseline Comparison**:
```
Config          Perplexity    FLOPs/token    Notes
---------------------------------------------------------
Fixed-1         ~120          1.0x           Fast, low quality
Fixed-2         ~100          1.5x           Balanced
Fixed-4         ~95           2.0x           Best quality, slow
Adaptive        ~96           1.2-1.4x       🎯 Sweet spot
```

---

## Experiment Configuration

### Default Settings (Balanced)

```python
# Model
hidden_dim = 512
num_layers = 6
num_heads = 8
ttt_layer_idx = 3  # Replace middle attention layer

# Training
num_epochs = 3
batch_size = 8
max_train_batches = 500  # ~4000 batches total, this is 1/8
max_eval_batches = 100
learning_rate = 3e-4

# Adaptive TTT
difficulty_metric = "entropy"
buckets = [1, 2, 4]
target_distribution = [0.3, 0.4, 0.3]
```

### Quick Test Settings (Fast)

For rapid iteration during development:

```bash
uv run python experiments/wikitext2_experiment.py \
    --mode all \
    --num_epochs 1 \
    --max_train_batches 50 \
    --max_eval_batches 10 \
    --device cuda
```

Runtime: ~10-15 minutes on GPU

### Full Settings (Publication Quality)

For best results:

```bash
uv run python experiments/wikitext2_experiment.py \
    --mode all \
    --num_epochs 5 \
    --max_train_batches 1000 \
    --max_eval_batches 200 \
    --device cuda
```

Runtime: ~8-10 hours on GPU

---

## Results Storage

All results are automatically saved:

```
experiments/
├── results/
│   ├── baseline_1_20251105_143022.json
│   ├── baseline_2_20251105_150415.json
│   ├── baseline_4_20251105_153808.json
│   └── adaptive_adaptive_20251105_161201.json
└── figures/
    ├── pareto_curve_wikitext2.png
    ├── allocation_distribution.png
    └── training_curves.png
```

**JSON Format**:
```json
{
  "config": "baseline",
  "ttt_iterations": 2,
  "num_params": 44122112,
  "test_perplexity": 98.45,
  "flops_per_token": 2.45e10,
  "ttt_stats": {
    "avg_iterations": 2.0,
    "flops_reduction": 0.0
  }
}
```

---

## Troubleshooting

### Out of Memory (GPU)

Reduce batch size or sequence length:

```bash
# Edit experiments/wikitext2_experiment.py
# Line ~250, change:
batch_size=4  # was 8
max_length=128  # was 256
```

### Slow Training

Check device is actually GPU:

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # GPU name
```

### Import Errors

Reinstall dependencies:

```bash
uv sync
```

### No Results Found

Check results directory exists:

```bash
ls experiments/results/
```

If empty, experiments haven't completed yet.

---

## Advanced: Custom Experiments

### Different TTT Layer Position

Edit `TransformerConfig`:
```python
config = TransformerConfig(
    ttt_layer_idx=0,  # Replace first layer
    # or
    ttt_layer_idx=5,  # Replace last layer
)
```

### Different Difficulty Metrics

Options: `"entropy"`, `"loss"`, `"gradient"`

```python
config = TransformerConfig(
    ttt_difficulty_metric="loss",  # Use loss-based
)
```

### Custom Iteration Buckets

```python
config = TransformerConfig(
    ttt_buckets=[1, 2, 3, 4],  # 4 levels
    ttt_target_distribution=[0.25, 0.25, 0.25, 0.25],
)
```

---

## Next Steps After Experiments

1. ✅ **Verify Results**: Check that metrics meet success criteria
2. 📊 **Document Findings**: Update README.md with results
3. 📈 **Analyze Patterns**: Look at which tokens get more/fewer iterations
4. 🔧 **Iterate**: Try different metrics, buckets, or layer positions
5. 📝 **Plan Week 2**: WikiText-103, ablations, optimizations

---

## Questions?

Check the detailed docs:
- [README.md](README.md) - Project overview
- [PLAN.md](PLAN.md) - Full 2-month timeline
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details

Happy experimenting! 🎉

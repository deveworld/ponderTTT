# PonderTTT Experiments

This directory contains all experiment scripts, results, and analysis tools.

---

## Files

### Main Experiments
- **`wikitext2_experiment.py`** - Run WikiText-2 language modeling experiments
  - Baselines: Fixed-1, Fixed-2, Fixed-4
  - Adaptive: Entropy-based allocation
  - Full training and evaluation pipeline

### Analysis
- **`analyze_wikitext2.py`** - Analyze and visualize results
  - Pareto curves (FLOPs vs Perplexity)
  - Allocation distributions
  - Training curves
  - Results tables

### Testing
- **`test_setup.py`** - Sanity checks to verify setup
  - Data loading
  - Model forward pass
  - Adaptive TTT functionality

### Legacy (Phase 1)
- `day3_analysis.py` - Phase 1 synthetic data analysis
- `toy_experiment.py` - Initial TTT experiments
- `visualize_results.py` - Phase 1 visualizations

---

## Quick Start

```bash
# 1. Test setup
uv run python experiments/test_setup.py

# 2. Run experiments (GPU recommended)
uv run python experiments/wikitext2_experiment.py --mode all --device cuda

# 3. Analyze results
uv run python experiments/analyze_wikitext2.py
```

---

## Directory Structure

```
experiments/
├── README.md                    # This file
├── wikitext2_experiment.py      # Main experiments
├── analyze_wikitext2.py         # Analysis & viz
├── test_setup.py                # Setup verification
├── results/                     # Auto-generated JSON results
│   ├── baseline_1_*.json
│   ├── baseline_2_*.json
│   ├── baseline_4_*.json
│   └── adaptive_*.json
└── figures/                     # Auto-generated plots
    ├── pareto_curve_wikitext2.png
    ├── allocation_distribution.png
    └── training_curves.png
```

---

## Expected Runtime

| Configuration | GPU | CPU |
|--------------|-----|-----|
| Quick test (50 batches, 1 epoch) | ~10 min | ~1 hour |
| Standard (500 batches, 3 epochs) | ~3 hours | ~20 hours |
| Full (1000 batches, 5 epochs) | ~8 hours | ~40+ hours |

**Recommendation**: Use GPU for experiments

---

## Results Format

Each experiment saves results as JSON:

```json
{
  "config": "baseline" | "adaptive",
  "ttt_iterations": 1 | 2 | 4 | "adaptive",
  "num_params": 44122112,
  "test_loss": 4.58,
  "test_perplexity": 97.23,
  "flops_per_token": 2.45e10,
  "ttt_stats": {
    "avg_iterations": 2.3,
    "flops_reduction": 0.325,
    "allocation_distribution": {
      "1": 0.30,
      "2": 0.42,
      "4": 0.28
    }
  }
}
```

---

## Next Steps

After completing WikiText-2 experiments:

1. **Week 2**: Scale to WikiText-103
2. **Week 3**: Optimization and profiling
3. **Week 4**: Penn Treebank validation
4. **Month 2**: Paper writing and arXiv submission

See [PLAN.md](../PLAN.md) for full timeline.

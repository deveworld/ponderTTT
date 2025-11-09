# PonderTTT - Status

**Last Updated**: 2025-11-09
**Phase**: Implementation Complete
**Status**: ✅ **ALL IMPLEMENTATION COMPLETE**
**Next**: Run Experimental Validation (when ready)

---

## 📌 Implementation

PonderTTT implements **adaptive iteration allocation for Test-Time Training**:
- Iterative gradient descent with K-step loops per token
- Learned halting policy using REINFORCE policy gradient
- Content-aware per-token allocation (configurable: per-token vs per-position)
- Fair comparison framework with parameter-matched baselines
- Model size: 60.10M trainable parameters (all methods matched)
  - All models include HaltingPolicyNetwork for parameter fairness
  - Baselines override policy with fixed num_steps

---

## ✅ Completed Components

### Core Models
- **IterativeTTTLayer**: K-step gradient descent with adaptive allocation
- **IterativeTTTLayerV2**: Configurable variants (linear/MLP, fixed/position-dependent LR)
- **OfficialTTTLayer**: Analytic baseline matching official TTT implementation
- **HaltingPolicyNetwork**: REINFORCE-based learned policy with per-token credit assignment
- **FastWeightModule**: MLP variant for gradient-based updates
- **LinearFastWeightModule**: Linear variant matching official TTT

### Heuristic Baselines
- **EntropyBasedPolicy**: Allocate based on prediction entropy
- **LossBasedPolicy**: Allocate based on per-token loss
- **GradientNormBasedPolicy**: Allocate based on gradient magnitude
- **PerplexityBasedPolicy**: Allocate based on local perplexity
- **RandomPolicy**: Random allocation (ablation baseline)
- **UniformPolicy**: Fixed K for all tokens
- **Calibration**: Percentile-based threshold calibration on validation set
  - More robust than per-batch min-max normalization
  - Consistent allocation across batches
  - Custom target distributions supported
  - 7 comprehensive tests - all passing ✅

### Experimental Framework
- **full_comparison_suite.py**: 8 methods with fair comparison (multi-seed support)
- **wikitext2_experiment.py**: Single experiment runner (for development)
- **convergence_analysis.py**: Iterative vs analytic gap analysis
- **oracle_analysis.py**: Optimal K allocation via exhaustive search
- **extended_oracle_analysis.py**: Extended oracle with visualization & Pareto frontier
  - Oracle K distribution plots
  - Difficulty-K correlation analysis
  - Oracle Pareto frontier (upper bound)
  - 6 comprehensive tests - all passing ✅

### Analysis Tools
- Convergence gap measurement (K-step vs analytic)
- Oracle K allocation (upper bound for learned policies)
- Difficulty-K correlation analysis
- Pareto frontier visualization
- Statistical significance testing

---

## 🚀 Experimental Methods

### 8 Comparison Methods
1. **uniform_k1**: Minimal compute baseline
2. **uniform_k2**: Low compute baseline
3. **uniform_k4**: Medium compute baseline
4. **uniform_k8**: Maximum compute baseline
5. **heuristic_entropy**: Entropy-based difficulty (non-learned)
6. **learned_lambda001_target4**: Main contribution (λ=0.01, target=4)
7. **learned_lambda005_target4**: Higher penalty (λ=0.05, target=4)
8. **learned_lambda001_notarget**: Minimize compute (λ=0.01, no target)

All methods use identical trainable parameter counts (60.10M) for fair comparison.

---

## 🎯 Research Objectives

### Primary Goals
- Investigate quality-compute tradeoffs via learned per-token iteration allocation
- Compare learned policies against uniform and heuristic baselines
- Validate that difficulty correlates with optimal K (oracle analysis)
- Demonstrate content-aware allocation improves over uniform allocation

### Evaluation Criteria
- Statistical significance: p < 0.05 with multiple seeds
- Fair comparison: All methods with matched parameter counts
- Accurate accounting: Exact FLOPs including all overheads
- Comprehensive analysis: Convergence, oracle bounds, correlations

---

## 📊 Key Features

### Novel Contributions
- Learned halting policy for TTT gradient descent
- REINFORCE-based end-to-end training with per-token credit assignment
- Content-aware per-token allocation (pooling='none')
- Quadratic compute penalty: λ(K - target)²

### Technical Implementation
- Configurable fast-weight architecture (linear/MLP)
- Position-dependent learning rate schedule
- Sequential and mini-batch processing variants
- Parameter fairness: all models with 60.10M trainable parameters

### Comparison Framework
- Official TTT analytic baseline
- Heuristic policies (entropy, loss, gradient)
- Uniform allocation baselines (K=1,2,4,8)
- Oracle allocation (exhaustive search)

---

## 📁 File Structure

```
src/ponderttt/
├── models/
│   ├── iterative_ttt.py              # Original iterative TTT
│   ├── iterative_ttt_v2.py           # Configurable variants
│   ├── ttt_linear_official.py        # Official analytic baseline
│   ├── fast_weight.py                # MLP fast-weight
│   ├── fast_weight_linear.py         # Linear fast-weight
│   ├── halting_policy.py             # REINFORCE policy
│   ├── heuristic_policies.py         # Non-learned baselines
│   ├── transformer_iterative.py      # Full transformer
│   └── __init__.py

├── experiments/
│   ├── full_comparison_suite.py      # 8 methods comparison
│   ├── convergence_analysis.py       # Iterative vs analytic
│   ├── oracle_analysis.py            # Optimal K allocation
│   ├── extended_oracle_analysis.py   # Extended oracle with visualization
│   └── wikitext2_experiment.py       # Single experiment

├── data/
│   └── wikitext.py                   # WikiText-2 loaders

└── utils/
    ├── metrics.py                    # Perplexity, FLOPs
    ├── statistics.py                 # Statistical tests
    ├── profiling.py                  # Performance analysis
    └── flops.py                      # Accurate FLOPs counting
```

---

## 🔬 Run Experiments

### Quick Validation (Development)
```bash
# Test 3 methods with 1 seed (~10 minutes)
python src/ponderttt/experiments/full_comparison_suite.py \
    --methods uniform_k1 uniform_k4 learned_lambda001_target4 \
    --seeds 42 \
    --num_epochs 1 \
    --max_train_batches 10 \
    --max_eval_batches 5 \
    --device cuda
```

### Full Comparison (Publication)
```bash
# All 8 methods with 5 seeds (5-7 days GPU)
python src/ponderttt/experiments/full_comparison_suite.py \
    --seeds 42 123 456 789 101112 \
    --num_epochs 10 \
    --device cuda
```

### Analysis Tools
```bash
# Convergence analysis (iterative vs analytic gap)
python src/ponderttt/experiments/convergence_analysis.py \
    --max_batches 50 \
    --k_values 1 2 4 8 16 \
    --device cuda

# Oracle analysis (optimal K per token)
python src/ponderttt/experiments/oracle_analysis.py \
    --max_batches 10 \
    --sample_positions 32 \
    --device cuda

# Extended oracle analysis (with visualization)
python src/ponderttt/experiments/extended_oracle_analysis.py \
    --max_batches 20 \
    --sample_positions 64 \
    --device cuda

# Single experiment (for development)
python src/ponderttt/experiments/wikitext2_experiment.py \
    --mode learned \
    --max_train_batches 100 \
    --max_eval_batches 20 \
    --device cuda
```

---

## ⏭️ Next Steps

### Immediate
1. **Run experiments**: Full comparison with 5+ seeds
2. **Generate figures**: Pareto curves, allocation distributions
3. **Statistical analysis**: Significance tests, effect sizes
4. **Validate oracle**: Confirm difficulty-K correlation

### Future Work
1. **Scale to WikiText-103**: Larger dataset validation
2. **Ablation studies**: Component contributions
3. **Wall-clock timing**: Real-world efficiency
4. **Bucketing optimization**: Parallel processing within K groups
5. **Publication**: Workshop or arXiv submission

---

See **README.md** for project details and architecture comparison.

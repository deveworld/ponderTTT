# PonderTTT

**Adaptive LR Scaling for Test-Time Training**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c.svg)](https://pytorch.org/)

## Quick Links

- [📊 Current Status](STATUS.md) - Implementation progress and next steps
- [🚀 Quick Start](QUICKSTART.md) - How to run experiments
- [📅 Development Plan](PLAN.md) - 2-month roadmap to publication

---

## Abstract

Test-Time Training (TTT) layers use gradient descent during inference to adapt model parameters to each input. Current methods apply a **fixed number of gradient steps** to all tokens, leading to inefficiency: easy tokens waste computation while hard tokens receive insufficient training.

**PonderTTT** introduces **difficulty-aware per-token iteration allocation** that dynamically adjusts the number of gradient steps K_t each token receives based on estimated difficulty. A learned halting policy network predicts optimal K_t using REINFORCE policy gradient for differentiable discrete decisions. While prior work (e.g., Adaptive Computation Time, Universal Transformers, PonderNet, DeltaProduct, MesaNet) studies compute/quality trade-offs in other settings, our focus is applying such policies directly to iterative gradient descent within TTT layers.

### Implementation

This implementation uses **iterative gradient descent TTT**:
- Explicit K-step gradient descent loops per token (not analytic solve)
- FastWeightModule: Small MLP for gradient-based parameter updates
- HaltingPolicyNetwork: Learned policy with REINFORCE policy gradient for differentiable K_t selection
- Model size: 60.10M total trainable parameters (all methods matched)
  - All models include HaltingPolicyNetwork for parameter fairness
  - Baselines override policy decisions with fixed num_steps
- Efficient implementation: Optimized for sequential test-time adaptation

---

## Problem & Solution

### Problem: Uniform Steps Waste Computation

```python
# Standard TTT: Fixed K steps for ALL tokens
for token in sequence:
    K_t = K_fixed  # Same for all
    for k in range(K_t):
        fast_weight = gradient_step(fast_weight, token)
    output = apply_fast_weight(token, fast_weight)
```

- **Easy tokens**: Converge in 1-2 steps → extra steps wasted
- **Hard tokens**: Need 8+ steps → K=4 insufficient
- **Result**: Suboptimal efficiency-quality tradeoff

### Solution: Learned Per-Token Iteration Allocation

```python
# PonderTTT: Learned halting policy
K_t = halting_policy(token_context)  # e.g., {1, 2, 4, 8}
for k in range(K_t):
    fast_weight = gradient_step(fast_weight, token)
output = apply_fast_weight(token, fast_weight)
```

**Key Innovation**:
- **Learned Halting Policy**: Neural network predicts optimal K_t per token
- **REINFORCE policy gradient**: Per-token credit assignment for end-to-end training
- **Per-Token Steps**: Each token receives different number of gradient steps
- **Parameter Fairness**: All models have 60.10M trainable parameters

---

## Status

**Implementation**: Complete ✅
**Validation**: Ready for WikiText-2 experiments

**Research Goals** (with 3 TTT layers):
- Investigate quality-compute tradeoffs through learned per-token iteration allocation
- Compare learned policy against uniform baselines at matched average compute budgets
- Report accurate per-token gradient steps and FLOPs using exact matmul cost (2·m·k·p)
- Validate results with ≥10 seeds for statistical significance

**Architecture Used**:
- **3 TTT layers** (layers 2, 3, 4) replacing attention
- **3 standard layers** with attention
- TTT layers: 22.7% of total FLOPs
- Standard layers: 22.0% of total FLOPs
- LM head: 55.3% of total FLOPs (dominant component)
- Adaptive overhead: 0.22% (difficulty estimation)

---

## Architecture Comparison with Official TTT

PonderTTT implements an **iterative variant** of TTT that enables adaptive per-token iteration counts. Key differences from the official implementation:

| Component | Official TTT | PonderTTT | Rationale |
|-----------|--------------|-----------|-----------|
| **Update Method** | Analytic (closed-form) | Iterative K-step GD | Enables variable K per token |
| **Fast Weight** | Linear (W @ K + b) | Configurable (linear/MLP) | Ablation-ready |
| **Processing** | Mini-batch (16 tokens) | Sequential per-token | Required for adaptive K |
| **Learning Rate** | Position-dependent | Configurable (fixed/position) | Simplified by default |
| **Steps per Token** | 1 analytic update | Variable: K ∈ {1,2,4,8} | Core innovation |

### Why Iterative Instead of Analytic?

The analytic TTT update computes a closed-form solution assuming:
- Fixed number of steps (single update per mini-batch)
- Parallel processing (all tokens together)

PonderTTT's **adaptive per-token allocation** requires:
- **Variable K_t**: Different tokens receive 1, 2, 4, or 8 gradient steps
- **Sequential processing**: To apply K_t iterations per token independently

This necessitates explicit gradient descent loops instead of analytic solutions.

### Novel Contributions vs Modifications

**Novel Contributions**:
- Learned halting policy for iteration allocation
- REINFORCE-based end-to-end training
- Content-aware difficulty metrics

**Modifications from Official TTT**:
- Iterative updates (necessary for variable K)
- Configurable fast-weight architecture (for ablation)
- Sequential processing (implementation detail)

**Fair Comparison**: We implement both variants for rigorous evaluation:
- `OfficialTTTLayer`: Analytic baseline matching official implementation
- `IterativeTTTLayerV2`: K-step iterative with all models at 60.10M trainable parameters

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run WikiText-2 experiments with statistical validation (FULL dataset, 10 epochs, 8 methods)
uv run python src/ponderttt/experiments/full_comparison_suite.py \
    --seeds 42 123 456 789 101112 999 888 777 666 555 \
    --num_epochs 10 \
    --device cuda

# Quick validation test (3 methods, 1 seed, ~1 hour)
uv run python src/ponderttt/experiments/full_comparison_suite.py \
    --methods uniform_k1 uniform_k4 learned_lambda001_target4 \
    --seeds 42 \
    --num_epochs 1 \
    --max_train_batches 10 \
    --max_eval_batches 5 \
    --device cuda

# Single experiment test (for development)
uv run python src/ponderttt/experiments/wikitext2_experiment.py \
    --mode learned --max_train_batches 100 --max_eval_batches 20 --device cuda
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

---

## Usage

```python
from src.ponderttt.models import IterativeTransformerTTT, IterativeTransformerConfig

# Configuration
config = IterativeTransformerConfig(
    vocab_size=50257,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    ttt_layer_indices=[2, 3, 4],  # Which layers use TTT
    max_steps=8,
    use_learned_policy=True,  # Use halting network
    step_options=[1, 2, 4, 8],  # K_t choices
)

# Create model
model = IterativeTransformerTTT(config)

# Forward pass with learned policy
outputs = model(
    input_ids=input_ids,
    labels=labels,
    num_steps=None,  # Let policy decide K_t
    return_stats=True,
)

# Check statistics
avg_steps = sum(s['avg_steps'] for s in outputs['ttt_stats']) / len(outputs['ttt_stats'])
print(f"Average gradient steps per token: {avg_steps:.2f}")
```

The `ttt_stats` field contains per-layer statistics including average steps and step distribution.

---

## Related Work

### Adaptive Computation in Neural Networks

PonderTTT builds on a rich history of adaptive computation methods, exploring a specific dimension: **per-token iteration allocation** (implemented as learned halting policy for gradient descent steps).

#### 1. Adaptive Forward Pass

**Adaptive Computation Time (ACT)** ([Graves 2016](https://arxiv.org/abs/1603.08983)) pioneered per-token variable computation for RNNs, using a halting unit to dynamically decide when to stop forward pass iterations. This has been extended to:

- **Universal Transformer** ([Dehghani et al. 2019](https://arxiv.org/abs/1807.03819)): ACT applied to Transformer layers
- **SELF-Transformer** ([arXiv:2507.13569](https://arxiv.org/abs/2507.13569)): Fixed-point iteration for attention refinement with convergence-based stopping
- **Mixture-of-Depths** ([Raposo et al. 2024](https://arxiv.org/abs/2404.02258)): Token routing to dynamically skip layers based on learned policies
- **Depth-Adaptive Transformers** ([Elbayad et al. 2020](https://arxiv.org/abs/1910.10073)): Early exiting from Transformer layers based on confidence

**Key Difference**: These methods adapt **forward pass depth** (how many layers to apply). PonderTTT adapts **gradient descent iterations** (how many steps K_t per token) within TTT layers.

#### 2. Test-Time Adaptation

**SIFT** ([OpenReview 2024](https://openreview.net/pdf?id=4Tlt7ANaLw)): Selects informative data for test-time fine-tuning using uncertainty-based sampling, then applies a **fixed single gradient step** per sample.

**LEAST** ([arXiv:2404.03784](https://arxiv.org/abs/2404.03784), AAAI 2025): Layerwise Early Stopping for TTA. Uses cosine distance-based criterion to **selectively stop adaptation of individual layers** when updates don't align with previous samples, preventing overfitting.

**Various TTA Methods** (TENT, SHOT, etc.): Adapt models at test-time using entropy minimization or self-supervised losses, but use **fixed adaptation steps** regardless of input difficulty.

**Key Differences**:
- **SIFT**: Data selection (which samples to use), not iteration control
- **LEAST**: Layer selection (which layers to adapt), gradient steps fixed per layer
- **PonderTTT**: Iteration allocation (how many gradient steps per token within each layer)

#### 3. Adaptive Computation in TTT-style Layers

Recent work has begun exploring dynamic computation in optimization-based recurrent layers:

**DeltaProduct** ([Siems et al., 2025](https://arxiv.org/abs/2501.16578)): Extends DeltaNet to multiple gradient steps per token (n_h steps → Householder product). Uses β coefficients to gate updates, allowing β=0 to skip steps. While this enables **implicit step control**, there is no explicit difficulty-based policy or FLOPs optimization objective. All tokens use the same n_h by default.

**MesaNet** ([Mesa layer, 2025](https://arxiv.org/abs/2501.16710)): Solves least-squares optimization per token using Conjugate Gradient with **dynamic stopping criterion** based on matrix conditioning. Adaptively allocates CG steps per token/head. However, the optimization objective differs (least-squares vs. self-supervised loss) and uses a linear system solver (CG) rather than gradient descent.

**Key Differences**:
- **DeltaProduct**: Multi-step gradient descent but implicit gating (β), no explicit halting policy
- **MesaNet**: Dynamic step allocation but for CG solver on least-squares, not gradient descent on self-supervised loss
- **PonderTTT**: Learned halting policy for gradient descent iterations with FLOPs-quality trade-off objective

#### 4. Recent TTT Advances

| Method | Focus | Relation to PonderTTT |
|--------|-------|----------------------|
| **Titans** (arXiv:2501.00663) | Surprise-weighted memory mechanism | Orthogonal (memory allocation vs compute allocation) |
| **MGG** (arXiv:2412.16901) | Learned optimizer for test-time adaptation | Compatible (gradient optimization vs iteration count) |

### PonderTTT's Contribution

**PonderTTT combines three key elements** not jointly addressed in prior work:

1. **Iterative gradient descent on self-supervised TTT loss** (explicit K-step loops, not analytic solve)
2. **Learned halting policy network** (REINFORCE policy gradient for differentiable K_t selection)
3. **FLOPs-quality trade-off optimization** (compute efficiency as primary objective)

**Comparison by adaptation dimension**:
- **ACT/Mixture-of-Depths**: "How many layers does this token need?" → Adapts **forward pass depth**
- **SIFT**: "Which data samples should I adapt on?" → Adapts **data selection**
- **LEAST**: "Which layers should I adapt?" → Adapts **layer selection**
- **DeltaProduct**: "How to expand expressive power?" → Adapts **parameter update richness** (implicit β gating)
- **MesaNet**: "When to stop CG solver?" → Adapts **linear system solver iterations**
- **PonderTTT**: "How many gradient steps for this token?" → Adapts **per-token iteration count K_t** (learned policy)

This approach is **complementary** to prior work:
- **ACT/Mixture-of-Depths + PonderTTT**: Hierarchical adaptation (layer routing + per-token iterations)
- **DeltaProduct + PonderTTT**: Expressive updates + learned halting
- **MesaNet insights**: Dynamic stopping criteria for different optimization types
- **SIFT + PonderTTT**: Smart data selection + adaptive iteration allocation
- **LEAST + PonderTTT**: Layer-wise early stopping + token-wise iteration control
- **MGG + PonderTTT**: Better gradient quality + adaptive step allocation
- **Titans + PonderTTT**: Memory-based context + efficient iteration allocation

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  IterativeTransformerTTT                     │
│  ┌────────────────────────────────────────┐  │
│  │ HaltingPolicyNetwork                   │  │
│  │  - Bi-LSTM context encoder             │  │
│  │  - Step predictor                      │  │
│  │  - REINFORCE policy gradient sampling             │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │ IterativeTTTLayer (per token)          │  │
│  │  - FastWeightModule (small MLP)        │  │
│  │  - Explicit K_t gradient steps         │  │
│  │  - create_graph=True for meta-learning │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │ Standard Transformer Layers            │  │
│  │  - Multi-head attention                │  │
│  │  - Feed-forward network                │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

---

## Experiments

### WikiText-2 Language Modeling

**Baseline Comparisons**:
- **Uniform-K**: All tokens receive same number of gradient steps (K=1,2,4,8)
- **Learned Policy**: HaltingPolicyNetwork predicts optimal K_t per token based on difficulty

**Metrics**:
- Test perplexity (quality) with bootstrap confidence intervals
- FLOPs per token (efficiency) based on actual gradient steps
- Per-token step distribution (K_t allocation)
- Statistical significance (10 seeds, t-test or Mann-Whitney U, Bonferroni correction)

**Run Commands**:
```bash
# Full comparison (all 8 methods, 10 seeds) - uses FULL WikiText-2 dataset
uv run python src/ponderttt/experiments/full_comparison_suite.py \
    --seeds 42 123 456 789 101112 999 888 777 666 555 \
    --num_epochs 10 \
    --device cuda

# Quick validation (3 methods, 1 seed, ~1 hour)
uv run python src/ponderttt/experiments/full_comparison_suite.py \
    --methods uniform_k1 uniform_k4 learned_lambda001_target4 \
    --seeds 42 \
    --num_epochs 1 \
    --max_train_batches 10 \
    --max_eval_batches 5 \
    --device cuda

# Development test (single experiment for debugging)
uv run python src/ponderttt/experiments/wikitext2_experiment.py \
    --mode learned --max_train_batches 100 --max_eval_batches 20 --device cuda
```

**Expected Runtime**:
- Quick validation: ~1 hour
- Full comparison: ~5-7 days on RTX 3090 (8 methods × 10 seeds × 10 epochs)

---

## Project Timeline

See [PLAN.md](PLAN.md) for 2-month roadmap to arXiv submission.

---

## Limitations

1. **Scale**: Current experiments use models with 512 hidden dimensions and 6 layers. Scalability to billion-parameter models remains to be validated.

2. **Domain**: Evaluation focuses on language modeling (WikiText-2). Generalization to other domains (vision, speech, reinforcement learning) is not explored.

3. **Overhead**: Adaptive allocation introduces computational overhead for difficulty estimation (0.22% of total FLOPs). The benefit comes from better quality-compute tradeoff, not raw FLOPs reduction.

4. **Iteration Allocation**: Current implementation varies the number of gradient steps K_t per token. FLOPs scale with K_t; efficiency comes from smart allocation (giving fewer steps to easy tokens).

### Addressed Limitations

- **Calibration Data**: Uses training set (separate from test evaluation)
- **TTT Variant**: Uses sequential TTT by default

---

## License

MIT License

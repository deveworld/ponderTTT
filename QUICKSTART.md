# PonderTTT Quick Start Guide

Get started with PonderTTT in 5 minutes!

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for experiments)
- [uv](https://github.com/astral-sh/uv) package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/ponderttt.git
cd ponderttt

# Install dependencies with uv
uv sync

# Verify installation
python -c "from src.ponderttt.models import IterativeTransformerTTT; print('✅ Installation successful!')"
```

---

## 🚀 Quick Validation (~1 hour)

Test the implementation with a minimal experiment:

```bash
uv run python src/ponderttt/experiments/full_comparison_suite.py \
    --methods uniform_k1 uniform_k4 learned_lambda001_target4 \
    --seeds 42 \
    --num_epochs 1 \
    --max_train_batches 10 \
    --max_eval_batches 5 \
    --device cuda
```

**What this does**:
- Compares 3 methods: Uniform-K1, Uniform-K4, Learned Policy
- Runs on 1 random seed
- Trains for 1 epoch on 10 batches
- Takes ~1 hour on RTX 3090

**Expected output**:
```
Method: uniform_k1
  Test Perplexity: ~150-200
  Avg Steps: 1.00

Method: uniform_k4
  Test Perplexity: ~100-120
  Avg Steps: 4.00

Method: learned_lambda001_target4
  Test Perplexity: ~105-125 (close to uniform_k4)
  Avg Steps: 2.5-3.5 (adaptive allocation)
```

---

## 🧪 Full Experiments (5-7 days)

Run complete experimental suite with statistical validation:

```bash
uv run python src/ponderttt/experiments/full_comparison_suite.py \
    --seeds 42 123 456 789 101112 999 888 777 666 555 \
    --num_epochs 10 \
    --device cuda
```

**What this does**:
- Compares all 8 methods (Uniform-K1/2/4/8, Learned policies, Heuristics)
- Runs 10 random seeds for statistical significance
- Trains for 10 epochs on full WikiText-2
- Generates comprehensive results

**Methods compared**:
1. `uniform_k1` - All tokens get 1 gradient step
2. `uniform_k2` - All tokens get 2 gradient steps
3. `uniform_k4` - All tokens get 4 gradient steps (standard)
4. `uniform_k8` - All tokens get 8 gradient steps
5. `heuristic_entropy` - Entropy-based allocation
6. `learned_lambda001_target4` - REINFORCE (λ=0.01, target=4) ⭐ Main contribution
7. `learned_lambda005_target4` - REINFORCE (λ=0.05, target=4)
8. `learned_lambda001_notarget` - REINFORCE (λ=0.01, minimize compute)

---

## 💻 Basic Usage

### Create and Train a Model

```python
from src.ponderttt.models import IterativeTransformerConfig, IterativeTransformerTTT
import torch

# Configure model
config = IterativeTransformerConfig(
    vocab_size=50257,           # GPT-2 tokenizer
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    ttt_layer_indices=[2, 3, 4],  # Which layers use TTT
    max_steps=8,
    use_learned_policy=True,    # Use REINFORCE policy
    step_options=[1, 2, 4, 8],  # K_t choices
)

# Create model
model = IterativeTransformerTTT(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 128))
labels = torch.randint(0, config.vocab_size, (2, 128))

outputs = model(
    input_ids=input_ids,
    labels=labels,
    num_steps=None,  # Let policy decide K_t
    return_stats=True,
)

# Check results
print(f"Loss: {outputs['loss'].item():.4f}")
if outputs['ttt_stats']:
    for i, stats in enumerate(outputs['ttt_stats']):
        print(f"TTT Layer {i}: avg_steps={stats['avg_steps']:.2f}")
```

### Load WikiText-2 Data

```python
from src.ponderttt.data.wikitext import get_wikitext2_dataloaders

train_loader, val_loader, test_loader = get_wikitext2_dataloaders(
    batch_size=8,
    max_length=256,
)

# Iterate over data
for batch in train_loader:
    input_ids = batch['input_ids']  # (batch, seq_len)
    labels = batch['labels']        # (batch, seq_len)
    break
```

### Compute FLOPs

```python
from src.ponderttt.utils.flops import compute_model_flops

flops = compute_model_flops(
    config,
    seq_len=256,
    num_steps=4,
    include_backward=True,
)

print(f"Total FLOPs: {flops['total']/1e9:.2f}G")
print(f"Per-token: {flops['per_token']/1e3:.2f}K")
print(f"Policy network overhead: {flops['policy_network']/1e6:.2f}M")
```

---

## 📊 Analysis Tools

### Oracle Analysis (Optimal K Allocation)

Find the optimal K for each token via exhaustive search:

```bash
uv run python src/ponderttt/experiments/oracle_analysis.py \
    --max_batches 20 \
    --sample_positions 64 \
    --device cuda
```

**Output**:
- Optimal K distribution
- Difficulty-K correlation
- Oracle Pareto frontier (upper bound)

### Convergence Analysis

Measure the gap between iterative and analytic TTT:

```bash
uv run python src/ponderttt/experiments/convergence_analysis.py \
    --max_batches 50 \
    --k_values 1 2 4 8 16 \
    --device cuda
```

**Output**:
- Convergence curves for different K
- Gap vs K plot
- Convergence rate analysis

### Extended Oracle with Visualization

```bash
uv run python src/ponderttt/experiments/extended_oracle_analysis.py \
    --max_batches 20 \
    --sample_positions 64 \
    --device cuda
```

**Output**:
- Oracle K distribution plots
- Difficulty-K correlation scatter plots
- Pareto frontier visualization

---

## 🔧 Configuration Options

### Model Configuration

```python
config = IterativeTransformerConfig(
    # Architecture
    vocab_size=50257,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    ffn_dim=2048,

    # TTT layers
    ttt_layer_indices=[2, 3, 4],  # Which layers use TTT
    fast_weight_hidden_dim=64,    # Fast-weight MLP hidden size
    ttt_base_lr=0.1,              # TTT learning rate
    max_steps=8,                  # Maximum gradient steps

    # Learned policy (REINFORCE)
    use_learned_policy=True,
    step_options=[1, 2, 4, 8],    # K_t choices
    lambda_compute=0.01,          # Compute regularization
    target_avg_steps=4.0,         # Target average steps
    gamma=0.99,                   # Discount factor
    baseline_momentum=0.99,       # Baseline EMA momentum
)
```

### Experiment Arguments

```bash
# full_comparison_suite.py arguments
--methods METHODS [METHODS ...]  # Methods to compare
--seeds SEEDS [SEEDS ...]        # Random seeds
--num_epochs NUM_EPOCHS          # Training epochs
--max_train_batches N            # Limit training batches (debug)
--max_eval_batches N             # Limit eval batches (debug)
--device {cuda,cpu}              # Device to use
--batch_size BATCH_SIZE          # Batch size (default: 8)
--max_length MAX_LENGTH          # Sequence length (default: 256)
```

---

## 📈 Expected Results

Based on our implementation (before running full experiments):

**Perplexity**:
- Uniform-K1: ~150-200 (low compute, poor quality)
- Uniform-K4: ~100-120 (medium compute, good quality)
- Uniform-K8: ~95-115 (high compute, best quality)
- **Learned Policy**: ~105-125 (medium-low compute, near-K4 quality) ⭐

**FLOPs Reduction**:
- Target: 15-30% reduction vs Uniform-K4
- Expected avg steps: 2.5-3.5 (adaptive allocation)

**Statistical Significance**:
- With 10 seeds: p < 0.05 (Bonferroni corrected)
- Bootstrap confidence intervals

---

## 🐛 Troubleshooting

### Import Errors

```bash
# If you get import errors:
python -c "import sys; print(sys.path)"

# Make sure you're in the project root and using:
uv run python ...
```

### CUDA Out of Memory

```bash
# Reduce batch size or sequence length:
--batch_size 4 --max_length 128
```

### Slow Training

```bash
# Use CPU for debugging (slower but works):
--device cpu

# Or reduce training size:
--max_train_batches 50 --max_eval_batches 10
```

---

## 📚 Next Steps

1. **Run Quick Validation** - Verify installation (~1 hour)
2. **Explore Code** - Read `src/ponderttt/models/transformer_iterative.py`
3. **Run Full Experiments** - Get publication-ready results (5-7 days)
4. **Analyze Results** - Use oracle and convergence analysis tools
5. **Read Paper Draft** - Understand the theoretical background

---

## 💡 Key Files

| File | Description |
|------|-------------|
| `README.md` | Project overview and architecture |
| `STATUS.md` | Implementation status and file structure |
| `PLAN.md` | Development roadmap and timeline |
| `src/ponderttt/models/transformer_iterative.py` | Main model implementation |
| `src/ponderttt/models/halting_policy.py` | REINFORCE policy network |
| `src/ponderttt/experiments/full_comparison_suite.py` | Main experiment script |
| `src/ponderttt/utils/flops.py` | Accurate FLOPs counting |

---

## 🆘 Getting Help

- **Issues**: Open an issue on GitHub
- **Questions**: Check README.md and STATUS.md
- **Code**: All code is documented with docstrings

**Happy experimenting!** 🚀

# PonderTTT

[Preprint on Web](https://ponderttt.worldsw.dev)

Adaptive, budget-aware Test-Time Training (TTT) for code generation models built with JAX/Flax NNX.

## Core Idea: Self-Supervised Adaptive Gating

PonderTTT introduces **Adaptive Test-Time Training (TTT)** with a fully self-supervised gating mechanism:

**TTT Reconstruction Loss** → Decides whether to update or skip.

This is **inference-compatible** because the gating signal (reconstruction loss) does not require ground-truth labels.

### How It Works

1. **Compute TTT Reconstruction Loss** $\mathcal{L}_{rec}$ for each input chunk (self-supervised).
2. **Gate Decision**: If $\mathcal{L}_{rec} > \tau$, perform TTT update. Otherwise, skip.
3. **Scale-Dependent Inversion**: 
   - **125M**: High $\mathcal{L}_{rec}$ → Update (model benefits from "hard" samples)
   - **350M+**: Low $\mathcal{L}_{rec}$ → Update (model destabilizes on "hard" samples)

### Verified Results

**Configuration**: 50% Update Budget, 1 Gradient Step per chunk.

| Model | Language | Baseline (SKIP) | Oracle | **Recon Gating** | **Oracle Capture** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **125M** | Python | 3.935 | 2.684 | **2.698** | **98.9%** |
| **125M** | JavaScript | 4.374 | 3.020 | **3.099** | **94.2%** |
| **125M** | Java | 4.927 | 3.344 | **3.403** | **96.3%** |
| **125M** | Go | 10.07 | 6.311 | **6.472** | **95.7%** |

> **Note**: 350M Python shows negative correlation (r=-0.60). Standard Recon Gating fails (3.78 > Random 3.27). **Inverted Gating** (update on low loss) is required for 350M+ models.

### Correlation: Reconstruction Loss vs Oracle Advantage

| Model | Language | **Recon Loss (Ours)** | TTT Improvement | Note |
| :--- | :--- | :--- | :--- | :--- |
| **125M** | Python | **+0.89** | +0.84 | Both positive |
| **125M** | Go | **+0.92** | +0.40 | Recon more robust |
| **350M** | Python | **-0.60** | -0.82 | **Inverted** |
| **350M** | Go | **-0.94** | -0.83 | **Inverted** |

> **Key Finding**: For 350M+ models, the correlation between reconstruction loss and Oracle Advantage is **negative**. This means high reconstruction loss actually *hurts* performance. We use **Inverted Gating** (update on *low* loss) for larger models.

## Technical Architecture

Pure JAX/Flax NNX implementation with multi-scale model support.

### Supported Models

| Model | Parameters | Status |
|-------|------------|--------|
| GPT-2 125M | 125M | ✅ Validated (Recon Gating) |
| GPT-2 350M | 350M | ✅ Validated (Inverted Gating) |
| GPT-2 Large | 774M | 🧪 Experimental |
| GPT-2 XL | 1.5B | 🧪 Experimental |
| Gemma 3 1B | 1B | In Progress |
| Gemma 3 4B | 4B | In Progress |
| Gemma 3 12B | 12B | In Progress (TPU) |

### Components

- **Base Model**: Pretrained backbone with frozen weights
- **TTT Layer**: Fast-weight adapter with self-supervised updates
- **Gating**: Training-free, self-supervised
  - **Reconstruction Gating**: Update when $\mathcal{L}_{rec} > \tau$ (125M) or $\mathcal{L}_{rec} < \tau$ (350M+)
  - Budget-aware threshold adjustment
  - Prediction entropy / Token confidence (Secondary signals)

### Loss Function

$$L_{total} = L_{CE} + \beta \cdot L_{TTT}$$

- $L_{CE}$: Main task cross-entropy (next-token prediction)
- $L_{TTT}$: TTT reconstruction loss (self-supervised adaptation signal, **also used for gating**)

## Installation

```bash
# Install uv if you do not have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project (CPU)
uv pip install -e .

# GPU (CUDA)
uv pip install -e . --group gpu

# TPU
uv pip install -e . --group tpu

# Development
uv pip install -e . --group dev
```

## Quick Start

### Reconstruction Gating (Inference-Compatible)

```python
# During inference, compute TTT reconstruction loss BEFORE deciding to update
output = model(input_ids, use_ttt=True)
recon_loss = output["ttt_stats"]["ttt_loss_step_0"]

# Gating decision (threshold calibrated from validation set)
if recon_loss > threshold:  # For 125M
    # Perform TTT update
    pass
else:
    # Skip update, use current weights
    pass

# For 350M+, invert the condition:
# if recon_loss < threshold: update
```

### Reproduce Paper Results

```bash
chmod +x scripts/run_all_experiments.sh

# Run all experiments (125M + 350M)
./scripts/run_all_experiments.sh

# Run specific model scales
./scripts/run_all_experiments.sh --125m          # 125M only
./scripts/run_all_experiments.sh --350m          # 350M only
./scripts/run_all_experiments.sh --1b            # GPT-2 Large (774M)
./scripts/run_all_experiments.sh --xl            # GPT-2 XL (1.5B)

# Run specific phases
./scripts/run_all_experiments.sh --125m phase1   # Training only
./scripts/run_all_experiments.sh --350m phase2   # Evaluation only

# Advanced options
./scripts/run_all_experiments.sh --350m phase2 --invert_signal  # Inverted gating for 350M+
./scripts/run_all_experiments.sh --1b phase2 --ttt_base_lr=0.1  # Custom learning rate
```

### Gemma 3 (TPU)

```python
from ponderttt.models.gemma3 import (
    Gemma3Config,
    Gemma3TTTModel,
    load_gemma3_from_huggingface,
    create_device_mesh,
    ShardingConfig,
)

# Initialize
config = Gemma3Config.gemma3_4b()
model = Gemma3TTTModel(config, ttt_config, rngs=rngs)

# Load pretrained weights
model = load_gemma3_from_huggingface(model, "google/gemma-3-4b-pt")

# Setup TPU sharding
mesh = create_device_mesh(ShardingConfig())
```

## Project Status

### Phase 1: Complete (Preprint)

- Pure NNX GPT-2, TTT Layer implementation
- Self-supervised Reconstruction Gating
- Results on GPT-2 (125M, 350M) with OOD evaluation
- Discovery: Scale-dependent correlation inversion

### Phase 2: In Progress

| Component | Status |
|-----------|--------|
| Reconstruction Gating | ✅ Complete |
| Inverted Gating (350M+) | ✅ Complete |
| Budget-aware Threshold | In Progress |
| Gemma 3 Integration | In Progress |
| TPU Pod Sharding | In Progress |
| LoRA-TTT | Planned |
| Reasoning Benchmarks | Planned |

## Repository Structure

```
ponderttt/
└── src/ponderttt/
    ├── models/
    │   ├── gpt2_nnx.py          # GPT-2 implementation
    │   ├── ttt_layer_nnx.py     # TTT layer
    │   └── gemma3/              # Gemma 3 (1B, 4B, 12B)
    │       ├── model.py         # Gemma3Model, Gemma3TTTModel
    │       ├── config.py        # Model configurations
    │       ├── checkpoint.py    # Weight loading
    │       └── sharding.py      # TPU Pod sharding
    ├── experiments/
    │   ├── train_baseline.py    # TTT baseline training
    │   ├── compare_methods.py   # Gating method comparison
    │   └── analyze_signals.py   # Signal correlation analysis
    └── data/
        └── pipeline.py          # Streaming data pipeline
```

## Citation

```bibtex
@article{sim2025ponderttt,
  title={Learning to Ponder: Adaptive Compute Allocation via Test-Time Training},
  author={Sim, Gihyeon},
  year={2025}
}
```

## License

MIT License

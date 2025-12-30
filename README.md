# PonderTTT

[Preprint on Web](https://ponderttt.worldsw.dev)

Adaptive, budget-aware Test-Time Training (TTT) for code generation models built with JAX/Flax NNX.

## Core Idea: Self-Supervised Adaptive Gating

PonderTTT introduces **Adaptive Test-Time Training (TTT)** with a fully self-supervised gating mechanism:

**TTT Reconstruction Loss** → Decides whether to update or skip.

This is **inference-compatible** because the gating signal (reconstruction loss) does not require ground-truth labels.

```mermaid
graph LR
    A[Input Chunk] --> B["TTT Forward (No Update)"]
    B --> C{Compute Full-Seq Recon Loss}
    C --> D{Check Threshold τ}
    D -- "L_rec > τ" --> E["UPDATE (Learn)"]
    D -- "L_rec ≤ τ" --> F["SKIP (Infer)"]
    E --> G[Update TTT State]
    G --> H[Final Forward Pass]
    F --> H
    H --> I[Next Token Prediction]
```

### How It Works

1. **Compute Full-Sequence Reconstruction Loss** $\mathcal{L}_{rec}$ for each input chunk (self-supervised).
2. **Gate Decision**: If $\mathcal{L}_{rec} > \tau$, perform TTT update. Otherwise, skip.
3. **Full-Seq Signal** (`ttt_loss_init`): Averages reconstruction loss across all positions in the chunk,
   providing stronger correlation with Oracle advantage than last-token-only loss.

### Verified Results

**Configuration**: 50% Update Budget, 1 Gradient Step per chunk.

| Model | Language | Baseline (SKIP) | Oracle | **Recon Gating** | **Oracle Recovery** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small (124M)** | Python | 2.324 | 2.006 | **1.994** | **103.6%** |
| **Small (124M)** | JavaScript | 3.164 | 2.461 | **2.343** | **116.5%** |
| **Small (124M)** | Java | 3.148 | 2.425 | **2.060** | **150.5%** |
| **Small (124M)** | Go | 6.130 | 4.189 | **4.053** | **107.0%** |
| **XL** | Python | 1.875 | 1.615 | **1.597** | **106.9%** |
| **XL** | JavaScript | 2.852 | 2.114 | **1.796** | **143.1%** |
| **XL** | Java | 3.213 | 2.268 | **2.057** | **122.3%** |
| **XL** | Go | 6.520 | 4.223 | **4.275** | **97.7%** |

> **Note**: Full-Sequence Reconstruction Gating achieves **>100% Oracle recovery** due to EMA-based thresholding, outperforming both Random and Oracle baselines.

### Correlation: Full-Sequence Reconstruction Loss vs Oracle Advantage

| Model | Language | **Pearson r** | Oracle Recovery |
| :--- | :--- | :--- | :--- |
| **Small (124M)** | Python | **0.84** | 103.6% |
| **XL** | Python | **0.61** | 106.9% |
| **XL** | JavaScript (OOD) | **0.74** | 143.1% |
| **XL** | Java (OOD) | **0.84** | 122.3% |
| **XL** | Go (OOD) | **0.58** | 97.7% |

> **Finding**: Small models show strong correlation ($r=0.84$), improving to moderate correlation at XL ($r=0.61$). Oracle Recovery consistently exceeds 100%.

## Technical Architecture

Pure JAX/Flax NNX implementation with multi-scale model support.

### Supported Models

| Model | Parameters | Status |
|-------|------------|--------|
| GPT-2 Small | 124M | ✅ Validated |
| GPT-2 Medium | 355M | ✅ Validated |
| GPT-2 Large | 774M | ✅ Validated |
| GPT-2 XL | 1.5B | ✅ Validated |
| Gemma 3 1B | 1B | In Progress |
| Gemma 3 4B | 4B | In Progress |
| Gemma 3 12B | 12B | In Progress (TPU) |
| Gemma 3 27B | 27B | In Progress (TPU) |

### Components

- **Base Model**: Pretrained backbone with frozen weights
- **TTT Layer**: Fast-weight adapter with self-supervised updates
- **Gating**: Training-free, self-supervised
  - **Reconstruction Gating**: Update when $\mathcal{L}_{rec} > \tau$
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

# GPU (CUDA 13)
uv pip install -e . --group gpu

# GPU (CUDA 12)
uv pip install -e . --group gpu12

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
if recon_loss > threshold:  # For GPT-2 Baseline
    # Perform TTT update
    pass
else:
    # Skip update, use current weights
    pass

# Gating decision based on reconstruction loss threshold
```

### Reproduce Paper Results

```bash
chmod +x scripts/run_all_experiments.sh

# Run all experiments (124M + 355M)
./scripts/run_all_experiments.sh

# Run specific model scales

./scripts/run_all_experiments.sh --small           # Small (124M) only
./scripts/run_all_experiments.sh --medium        # Medium (355M) only
./scripts/run_all_experiments.sh --large         # Large (774M) only

# Or run specific phases
./scripts/run_all_experiments.sh --small phase1   # Training only
./scripts/run_all_experiments.sh --medium phase2   # Evaluation only

# Run with custom hyperparameters
./scripts/run_all_experiments.sh --large phase2 --ttt_base_lr=0.1  # Custom learning rate
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
- Results on GPT-2 (Small, Medium, Large, XL) with OOD evaluation
- Finding: Reconstruction gating provides marginal improvement over random selection

### Phase 2: In Progress

| Component | Status |
|-----------|--------|
| Reconstruction Gating | ✅ Complete |
| Budget-aware Threshold | In Progress |
| Learned Gating Signals | In Progress |
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
    │   └── gemma3/              # Gemma 3 (1B, 4B, 12B, 27B)
    │       ├── model.py         # Gemma3Model, Gemma3TTTModel
    │       ├── config.py        # Model configurations
    │       ├── checkpoint.py    # Weight loading
    │       └── sharding.py      # TPU Pod sharding
    ├── experiments/
    │   ├── train_baseline.py    # TTT baseline training
    │   ├── compare_methods.py   # Gating method comparison
    │   └── analyze_signals.py   # Signal correlation analysis
    └── data/
        └── dataset.py           # Streaming data pipeline
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

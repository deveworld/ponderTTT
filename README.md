# PonderTTT

[Preprint on Web](https://ponderttt.worldsw.dev)

Adaptive, budget-aware Test-Time Training (TTT) for code generation models built with JAX/Flax NNX.

## Core Idea: Training-Free Adaptive Gating

PonderTTT introduces **Adaptive Test-Time Training** with learned SKIP/UPDATE decisions. We developed a **Crawl-Walk-Run** approach that achieves 80% of oracle performance without any additional training:

| Phase | Method | Oracle Capture | Online | Training |
|-------|--------|----------------|--------|----------|
| Crawl | Top-k TTT Improvement | 80.5% | No | None |
| Walk | Fixed Threshold | 69.3% | Yes | None |
| Run | Multi-signal + Budget-aware | TBD | Yes | Optional |

**Key Insight**: TTT's internal self-supervision loss directly measures "how much the model wants to learn" from the current context (Spearman ρ = 0.63 with oracle).

### Key Results (GPT-2 125M on Python)

| Method | Loss | Cost | vs Random |
|--------|------|------|-----------|
| Random Skip (50%) | 3.619 | 2.0x | baseline |
| **TTT Improvement** | **3.307** | **2.0x** | **+8.6%** |
| UPDATE_1 (always) | 3.328 | 3.0x | - |
| Oracle (upper bound) | 3.231 | 2.0x | +10.7% |

**TTT Improvement gating beats always-UPDATE with 33% less compute!**

### OOD Generalization (trained on Python, evaluated on 1000 chunks)

| Language | Baseline (SKIP) | UPDATE_1 | Improvement |
|----------|-----------------|----------|-------------|
| JavaScript | PPL 120 | PPL 56 | 2.1x |
| Java | PPL 162 | PPL 63 | 2.6x |
| Go | PPL 13,243 | PPL 1,672 | 7.9x |

Note: Go's high baseline PPL reflects GPT-2's weak performance on Go syntax.

## Technical Architecture

Pure JAX/Flax NNX implementation with multi-scale model support.

### Supported Models

| Model | Parameters | Status |
|-------|------------|--------|
| GPT-2 125M | 125M | Validated |
| GPT-2 350M | 350M | Validated |
| Gemma 3 1B | 1B | In Progress |
| Gemma 3 4B | 4B | In Progress |
| Gemma 3 12B | 12B | In Progress (TPU) |

### Components

- **Base Model**: Pretrained backbone with frozen weights
- **TTT Layer**: Fast-weight adapter with self-supervised updates
- **Gating**: Training-free (TTT Improvement) or learned (Multi-signal)
  - TTT Improvement signal (ρ = 0.63 correlation with oracle)
  - Prediction entropy
  - Token confidence
  - Budget-aware threshold adjustment

### Loss Function

$$L_{total} = L_{CE} + \beta \cdot L_{TTT}$$

- $L_{CE}$: Main task cross-entropy (next-token prediction)
- $L_{TTT}$: TTT reconstruction loss (self-supervised adaptation signal)

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

### Training-Free Gating (Recommended)

```python
from ponderttt.models import create_advanced_gating

# Create gating network
gating = create_advanced_gating(
    mode="threshold",  # or "budget_aware", "learned"
    use_entropy=True,
    use_token_confidence=True,
    target_update_rate=0.5,
)

# Get gating decision
result = gating(
    ttt_improvement=ttt_stats["ttt_loss_step_0"] - ttt_stats["ttt_loss_step_1"],
    logits=model_output["logits"],
)
should_update = result["decision"]
```

### Reproduce Paper Results

```bash
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh
```

### Evaluate Gating Methods

```bash
# Compare TTT-only vs Multi-signal vs Budget-aware
python -m ponderttt.experiments.evaluate_advanced_gating \
    --checkpoint outputs/baselines/125m_update1/checkpoints/checkpoint_100000/ \
    --num_batches 1000
```

### Train Fixed Baselines

```bash
python -m ponderttt.experiments.train_hard_skip \
    --model_scale 125m \
    --target_update_rate 0.5 \
    --num_iterations 10000 \
    --output_dir outputs/hard_skip
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
- Gumbel-Softmax training for SKIP/UPDATE decisions
- End-to-End differentiable training with budget constraints
- Results on GPT-2 (125M, 350M) with OOD evaluation

### Phase 2: In Progress

| Component | Status |
|-----------|--------|
| Crawl Phase (TTT Improvement gating) | Complete |
| Walk Phase (Fixed threshold, online) | Complete |
| Run Phase (Multi-signal + budget-aware) | In Progress |
| Gemma 3 Integration (1B, 4B, 12B) | In Progress |
| TPU Pod Sharding | In Progress |
| LoRA-TTT | Planned |
| Reasoning Benchmarks (MATH500, GSM8K, etc.) | Planned |

## Repository Structure

```
ponderttt/
└── src/ponderttt/
    ├── models/
    │   ├── gpt2_nnx.py          # GPT-2 implementation
    │   ├── ttt_layer_nnx.py     # TTT layer
    │   ├── advanced_gating.py   # Multi-signal gating (Run phase)
    │   └── gemma3/              # Gemma 3 (1B, 4B, 12B)
    │       ├── model.py         # Gemma3Model, Gemma3TTTModel
    │       ├── config.py        # Model configurations
    │       ├── checkpoint.py    # Weight loading
    │       └── sharding.py      # TPU Pod sharding
    ├── experiments/
    │   ├── train_hard_skip.py   # Main training script
    │   ├── compare_methods.py   # Baseline comparison
    │   └── evaluate_advanced_gating.py  # Run phase evaluation
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

# PonderTTT: Adaptive Budget-Constrained Test-Time Training

Learning When to Ponder During Test-Time Training for Code Generation

**JAX/Flax Implementation for TPU Optimization**

## Overview

PonderTTT is a reinforcement learning framework for adaptive test-time training (TTT) that learns **when** to perform parameter updates during inference. Unlike existing TTT methods that use fixed update schedules, PonderTTT uses a learned policy to dynamically allocate computational budget across input chunks.

### Key Features

- **JAX/Flax Implementation**: Optimized for TPU v4-64
- **Adaptive Computation**: Learns to allocate TTT updates based on chunk difficulty
- **Budget-Constrained**: Uses PID-Lagrangian PPO to enforce computational budgets
- **Code-Specific**: Optimized for code generation with domain-specific features
- **Efficient**: Target 30-40% computational savings vs fixed-schedule baselines

## Architecture Stack

```
JAX/Flax Stack:
├── JAX 0.8.0+            # Numerical computing
├── Flax 0.12.0+          # Neural network library
├── Optax 0.2.6+          # Optimization
├── Orbax                 # Checkpointing
├── Transformers 4.57.1+  # Pre-trained models
└── Datasets              # Data loading
```

**Note on JAX Version**: Minimum JAX 0.8.0 required. The codebase uses modern JAX patterns (NamedSharding, mesh_utils, jax.make_mesh). We specify 0.8.0+ for TPU compatibility while supporting newer versions.

## Installation

### Prerequisites

This project uses **uv** for dependency management (not pip). Install uv first:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### For TPU (Future - Not Yet Tested)

```bash
# On TPU VM
uv pip install -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    jax[tpu]==0.8.0

# Install other dependencies
uv pip install -e .
```

**Note**: TPU support is implemented but not yet validated on actual hardware.

### For GPU (Recommended for Production)

```bash
# Install JAX for GPU
uv pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
uv pip install -e .
```

**Note**: GPU is required for production experiments. CPU is too slow for real training.

### For CPU (Development/Validation Only)

```bash
uv pip install -e .
```

**Note**: CPU validation confirmed working. Use for development and testing only.

## Quick Start

### 1. Run Baseline Experiments (Requires GPU)

```bash
# Run baseline with The Stack dataset
uv run python -m ponderttt.experiments.train_baseline \
    --model_scale 125m \
    --action UPDATE_1 \
    --max_chunks 100
```

This runs:
- JAX/Flax pipeline
- Model initialization (GPT-2 125M)
- TTT layer with fast weights
- Data loading from The Stack v2
- Training with fixed action schedule

**Available Actions**:
- `--action SKIP`: No TTT updates (1× cost)
- `--action UPDATE_1`: 1 TTT step (3× cost)
- `--action UPDATE_2`: 2 TTT steps (6× cost)
- `--action UPDATE_4`: 4 TTT steps (12× cost)

**Note**: GPU required for production training. CPU is too slow for real experiments.

### 2. Test Distributed Setup (Single Host)

```bash
python scripts/test_distributed.py
```

This tests:
- JAX distributed initialization
- Mesh creation
- Data sharding
- Collective operations

### 2.5. Validate TPU Setup (Recommended)

Before running expensive training jobs, validate your TPU setup:

```bash
# Single host
python scripts/validate_tpu_setup.py

# Multi-host (run on ALL hosts)
gcloud compute tpus tpu-vm ssh ponderttt-v4-64 \
  --zone=us-central2-b \
  --worker=all \
  --command="cd ~/ponderttt && python scripts/validate_tpu_setup.py --multi_host"
```

This comprehensive validation checks:
- JAX distributed initialization
- Device mesh creation
- Data and parameter sharding
- Gradient computation and aggregation
- Cross-host collective operations
- JIT compilation with sharding

### 3. TPU Setup (for actual experiments)

#### Single Host (TPU v4-8)

```bash
# Create TPU VM
gcloud compute tpus tpu-vm create ponderttt-v4-8 \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base

# SSH into TPU
gcloud compute tpus tpu-vm ssh ponderttt-v4-8 --zone=us-central2-b

# Clone and setup
git clone <your-repo>
cd ponderttt
pip install -e .

# Test distributed setup
python scripts/test_distributed.py

# Run training
python scripts/train_tpu.py --mesh_shape="8,1" --num_steps=1000
```

#### Multi-Host (TPU v4-64: 8 hosts)

```bash
# Create TPU Pod
gcloud compute tpus tpu-vm create ponderttt-v4-64 \
  --zone=us-central2-b \
  --accelerator-type=v4-64 \
  --version=tpu-ubuntu2204-base

# Setup on ALL hosts (runs command on all 8 hosts)
gcloud compute tpus tpu-vm ssh ponderttt-v4-64 \
  --zone=us-central2-b \
  --worker=all \
  --command="git clone https://github.com/deveworld/ponderttt.git && cd ponderttt && pip install -e ."

# Test distributed setup (on all hosts)
gcloud compute tpus tpu-vm ssh ponderttt-v4-64 \
  --zone=us-central2-b \
  --worker=all \
  --command="cd ponderttt && python scripts/test_distributed.py --multi_host"

# Run training (on all hosts simultaneously)
gcloud compute tpus tpu-vm ssh ponderttt-v4-64 \
  --zone=us-central2-b \
  --worker=all \
  --command="cd ponderttt && python scripts/train_tpu.py --multi_host --mesh_shape='64,1' --global_batch_size=512"
```

**Important**:
- For multi-host, the training script MUST run on ALL hosts simultaneously
- JAX automatically coordinates between hosts using `jax.distributed.initialize()`
- Each host processes its own shard of data

## Project Structure

```
ponderttt/
├── src/ponderttt/
│   ├── data/              # Data loading (The Stack)
│   │   ├── dataset.py     # CodeDataset, data iterators
│   │   └── tokenization.py
│   ├── models/            # Flax models
│   │   ├── base_model.py  # TransformerLM (HF wrapper)
│   │   ├── ttt_layer.py   # TTT layer with fast weights (primary)
│   │   ├── policy.py      # PolicyNetwork (actor-critic)
│   │   └── fast_weights.py # LoRA-style fast weights (alternative)
│   ├── training/          # Training algorithms
│   │   ├── pid_lagrangian.py # PID-Lagrangian PPO
│   │   ├── ttt_trainer.py    # TTT baseline trainer
│   │   └── policy_trainer.py # Policy trainer
│   └── utils/             # Utilities
│       ├── features.py    # 32D feature extraction
│       ├── statistics.py  # Statistical tools
│       └── checkpointing.py # Orbax checkpointing
├── scripts/
│   └── quick_test.py      # Quick verification
└── PLAN.md               # Research plan v2.0
```

## Implementation Details

### Action Space

|  Action  | TTT Steps | Cost |       Use Case      |
|----------|-----------|------|---------------------|
| SKIP     |     0     |  1×  | Easy chunks         |
| UPDATE_1 |     1     |  3×  | Moderate difficulty |
| UPDATE_2 |     2     |  6×  | Difficult chunks    |
| UPDATE_4 |     4     |  12× | Very difficult      |

### Feature Space (32D)

Policy uses 32-dimensional features:
1. **Model Confidence** (4D): Entropy, perplexity
2. **Activations** (6D): Mean, std, sparsity, range
3. **Attention** (4D): Entropy, range, sparsity
4. **Code Metrics** (8D): Token stats, diversity
5. **History** (4D): EMA tracking, budget remaining
6. **Sequence** (6D): Length, frequency, compression

### Fast Weights

Inspired by LaCT's SwiGLU fast weights:
```python
# Fast weight function
y = w1 @ (silu(w0 @ x) * (w2 @ x))

# Chunked processing (4096 tokens/chunk)
# Apply then Update paradigm
```

## Key Differences from PyTorch Version

### Why JAX/Flax?

1. **TPU Optimization**: JAX is designed for TPUs
2. **Functional Programming**: Clean gradient computation
3. **XLA Compilation**: Better performance at scale
4. **Vmap/Pmap**: Easy parallelization
5. **Research Community**: TTT-LM uses JAX

### For PyTorch Users

JAX/Flax equivalents to common PyTorch patterns:

- **nn.Module → Flax Module**: Similar class-based style but functional
- **torch.optim → Optax**: Functional optimizers
- **DataLoader → Iterator**: JAX-friendly data pipeline
- **.to(device) → jit/pmap**: JAX handles devices differently

## Comparison with Related Work

### vs. TTT-LM-JAX
- **TTT-LM**: Fixed TTT architecture
- **PonderTTT**: Learned adaptive policy on top

### vs. LaCT
- **LaCT**: PyTorch, fixed fast-weight updates
- **PonderTTT**: JAX/Flax, RL-based adaptive updates

## Roadmap

See [PLAN.md](PLAN.md) for detailed research plan and [PROJECT_STATUS.md](PROJECT_STATUS.md) for current status.

### Phase 1: Foundation COMPLETE
- JAX/Flax implementation
- Data pipeline with multi-host sharding
- Core models (TTT, Policy)
- Training algorithms (PID-Lagrangian PPO)
- Feature extraction
- Multi-host distributed training
- TPU-ready training scripts (not yet tested on hardware)
- CPU validation complete
- Bug fixes (chunk size, dropout, dynamic slicing, etc.)
- Cost calculations validated (SKIP=1×, UPDATE_1=3×, UPDATE_2=6×, UPDATE_4=12×)

### Phase 2: Real Data & GPU (Current - Partially Blocked)
- **UNBLOCKED**: The Stack v2 dataset access approved
- **Blocker**: GPU access for production training (CPU too slow)
- [ ] 125M baseline experiments with real data
- [ ] Policy training and evaluation
- [ ] Ablation studies
- [ ] 350M scaling

### Phase 3: Publication (Future)
- [ ] 1B experiments
- [ ] Statistical analysis
- [ ] Paper writing

## References

**Code Bases**:
- [TTT-LM-JAX](https://github.com/test-time-training/ttt-lm-jax) - TTT architecture
- [LaCT](https://github.com/a1600012888/LaCT) - Fast weight concepts

**Papers**:
- TTT-LM: "Learning to (Learn at Test Time)", arXiv:2407.04620
- LaCT: "Test-Time Training Done Right", arXiv:2505.23884
- PID-Lagrangian: "Responsive Safety in RL", ICML 2020

## License

MIT License - see LICENSE file

## Citation

```bibtex
@software{ponderttt2025,
  title={PonderTTT: Adaptive Budget-Constrained Test-Time Training},
  author={deveworld},
  year={2025},
  note={JAX/Flax implementation}
}
```

## Contact

For questions, open an issue on GitHub.

---

**Version**: 0.2.0 (JAX/Flax)
**Status**: Pipeline Validated on CPU, Ready for GPU/Real Data

## Current Limitations

### Known Issues
1. **GPU Required**: CPU validation confirms pipeline works, but GPU needed for production experiments (CPU too slow).
2. **Real Benchmarks Pending**: HumanEval/MBPP evaluation requires real code training data.

### Implementation Notes (v0.2.0)
1. **Model Integration**: HuggingFace/Flax compatibility wrapper for seamless model loading
2. **Training Mode**: Proper RNG handling for dropout layers
3. **JAX Compatibility**: Dynamic slicing in TTT layer for flexible sequence handling
4. **FSDP Strategy**: Fully Sharded Data Parallel for memory-efficient training on large models

### Next Steps
1. Secure GPU resources for training (Vast.ai / RunPod)
2. Run baseline experiments with real code data
3. Validate TTT improvement on meaningful data

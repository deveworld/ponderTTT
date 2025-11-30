# PonderTTT

[[Preprint]](./PonderTTT_preprint.pdf)

Adaptive, budget-aware Test-Time Training (TTT) for code generation models built with JAX/Flax NNX.

## Core Idea: Binary Gating via Gumbel-Softmax

PonderTTT introduces **Adaptive Test-Time Training** with learned SKIP/UPDATE decisions. Instead of applying TTT updates uniformly to all input chunks, we learn **when** to update using a binary gating mechanism trained via Gumbel-Softmax.

| Feature | Fixed TTT | PonderTTT (Binary Gating) |
| :--- | :--- | :--- |
| **Decision** | Always UPDATE | SKIP or UPDATE per chunk |
| **Training** | N/A | Gumbel-Softmax (differentiable) |
| **Inference** | Fixed cost | True computational savings |
| **Cost** | 3.0x (UPDATE_1) | 2.67x (83% update rate) |

**Key Results (GPT-2 125M on Python):**
- 4.5x perplexity improvement over non-adaptive baseline (26.36 → 5.85)
- Strong OOD generalization: JavaScript (2.5x), Java (6.2x), Go (70x)
- Learned policy captures universal "when to adapt" patterns

## Technical Architecture

This project is a pure JAX/Flax NNX rewrite of the official TTT-LM, enhanced with adaptive gating.

- **Base Model**: Pretrained GPT-2 (125M, 350M) with frozen backbone weights
- **Fast-Weight Layer (`TTTLayer`)**: TTT-Linear with causal convolutions and dual-form updates
- **Binary Gating Network**: Lightweight MLP that makes SKIP/UPDATE decisions via Gumbel-Softmax
- **End-to-End Loss**:
  $$L_{total} = L_{CE} + \beta \cdot L_{TTT} + \gamma \cdot L_{cost}$$
  - $L_{CE}$: Main task cross-entropy
  - $L_{TTT}$: TTT reconstruction loss
  - $L_{cost}$: Penalty for computational budget (encourages skipping)

## Installation

```bash
# Install uv if you do not have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project in editable mode (CPU only)
uv pip install -e .
# GPU
uv pip install -e . --group gpu
# TPU
uv pip install -e . --group tpu

# Optional extras
uv pip install -e . --dev
```

## Quick Start

### 1. Validate Setup
Run the lightweight smoke tests to ensure tokenizer and model loading work:
```bash
python scripts/quick_test.py
python scripts/test_pipeline.py
```

### 2. Binary Gating Training (Main Workflow)
Train the binary gating network with Gumbel-Softmax on The Stack v2:
```bash
python -m ponderttt.experiments.train_hard_skip \
    --model_scale 125m \
    --target_skip_rate 0.5 \
    --num_iterations 10000 \
    --output_dir outputs/hard_skip
```
Checkpoints will be saved to `outputs/hard_skip` (e.g., `checkpoint_10000`).

### 3. Compare Methods
Evaluate trained models against fixed baselines:
```bash
python -m ponderttt.experiments.compare_methods \
    --model_scale 125m \
    --budget 2.0 \
    --num_eval_batches 20
```

### 4. Evaluation (Benchmarks)
Use `ponderttt.evaluation.benchmarks` for HumanEval/MBPP. Code execution is unsafe and gated by `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS=1`. Only set this in a sandboxed environment.

### 5. Checkpointing
Models are saved using [Orbax](https://github.com/google/orbax). Checkpoints capture the full NNX state, including the Gating Network, TTT parameters, and optimizer state.

## Project Status

### Phase 1: Complete (Preprint, arXiv TBD)
- Pure NNX GPT-2, TTT Layer with Binary Gating
- Gumbel-Softmax training for SKIP/UPDATE decisions
- End-to-End differentiable training with budget constraints
- Results on GPT-2 (125M, 350M) with OOD evaluation

### Phase 2: Planned (Conference Submission)
- Scale to Gemma 3 (4B, 12B)
- LoRA-TTT for efficiency
- Reasoning benchmarks: MATH500, GSM8K, LiveCodeBench, GPQA-Diamond
- Advanced gating features: Entropy, VOG, Attention Dispersion

See `PLAN.md` for the detailed research roadmap.

## Citation

```bibtex
@article{sim2025ponderttt,
  title={Learning to Ponder: Adaptive Compute Allocation via Test-Time Training},
  author={Sim, Gihyeon},
  year={2025}
}
```

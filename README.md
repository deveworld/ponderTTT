# PonderTTT

Adaptive, budget-aware Test-Time Training (TTT) for code generation models built with JAX/Flax NNX.

## Core Idea: Differentiable Continuous Gating

PonderTTT introduces **Differentiable Adaptive Test-Time Training**. Instead of using fixed update schedules or unstable Reinforcement Learning (RL), we learn **how strongly** to update fast weights during inference using a continuous, gradient-based approach.

| Feature | Discrete (RL/PPO) | PonderTTT (Continuous Gating) |
| :--- | :--- | :--- |
| **Control Signal** | Action $a_t ∈ \{0, 1, 2, 4\}$ | Scalar $\lambda_t \in [0, n]$ |
| **Update Logic** | Select $k$ distinct gradient steps | Scale the learning rate $\eta$ by $\lambda_t$ |
| **Optimization** | Policy Gradient (High Variance) | End-to-End Backprop (Stable) |
| **Benefit** | Hard exploration | Smooth loss landscape, "Soft Skips" |

This allows the model to allocate compute resources optimally—spending more updates on difficult code chunks and saving on boilerplate—while being trained efficiently via standard backpropagation.

## Technical Architecture

This project is a pure JAX/Flax NNX rewrite of the official TTT-LM, enhanced with adaptive capabilities.

- **Base Model**: Pretrained GPT-2 (125M~1.5B) implementation in NNX. The backbone weights are frozen (`stop_gradient`) during fine-tuning.
- **Fast-Weight Layer (`TTTLayer`)**: Implements TTT-Linear with causal convolutions and dual-form updates. We extended the update rule to accept a `gating_scale` ($\lambda_t$), effectively scaling the learning rate element-wise.
- **Gating Network**: A lightweight MLP that observes 32-D features (loss, entropy, budget remaining) and predicts $\lambda_t$.
- **End-to-End Loss**:
  $$L_{total} = L_{CE} + \beta \cdot L_{TTT} + \gamma \cdot \text{Mean}(\lambda_t)$$
  - $L_{CE}$: Main task cross-entropy.
  - $L_{TTT}$: Reconstruction loss of the TTT layer (ensures fast weights learn useful representations).
  - $\gamma \cdot \text{Mean}(\lambda_t)$: Soft penalty to enforce the compute budget.

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

### 2. Differentiable Training (Fine-tuning)
This is the main workflow. Train the gating network and TTT parameters jointly on The Stack v2:
```bash
python -m ponderttt.experiments.train_differentiable \
    --model_scale 125m \
    --max_steps 4.0 \
    --budget_limit 2.0 \
    --num_iterations 1000 \
    --output_dir outputs/differentiable
```
Checkpoints will be saved to `outputs/differentiable` (e.g., `checkpoint_1000`).

### 3. Compare Methods
Evaluate your trained model against fixed baselines and RL (PPO) approaches. You can load trained checkpoints for evaluation:
```bash
python -m ponderttt.experiments.compare_methods \
    --model_scale 125m \
    --budget 2.0 \
    --num_eval_batches 20 \
    --diff_checkpoint outputs/differentiable/checkpoint_1000 \
    --rl_checkpoint outputs/policy_nnx/seed_42/checkpoint_100
```

### 4. Evaluation (Benchmarks)
Use `ponderttt.evaluation.benchmarks` for HumanEval/MBPP/ClassEval. Code execution is unsafe and gated by `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS=1`. Only set this in a sandboxed environment.

### 5. Checkpointing
Models are saved using [Orbax](https://github.com/google/orbax). Checkpoints capture the full NNX state, including the Gating Network, TTT parameters, and optimizer state.

## Project Status
- **Complete**: Pure NNX GPT-2, TTT Layer with Continuous Gating, End-to-End Differentiable Training Loop, Budget-Awareness, Comparison Script, Checkpointing (Orbax), Benchmarks (HumanEval, MBPP, ClassEval).
- **In Progress**: Large-scale fine-tuning experiments and OOD (Out-of-Distribution) testing.

See `PLAN.md` for the detailed research roadmap.

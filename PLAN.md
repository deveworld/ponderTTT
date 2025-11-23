# PonderTTT Research Plan

## 1. Problem statement
We study adaptive test-time training for code generation models. A pretrained GPT-2 (125 M–1 B parameters) is frozen; a fast-weight adapter (TTT layer) can be updated after each chunk of generated code. We must decide **how intensely** to update the adapter to minimize loss while respecting a compute budget. We propose a **differentiable, continuous gating** approach that outperforms discrete RL-based control in stability and implementation simplicity.

## 2. Method overview
1. **Chunked streaming** – The Stack v2 (Python) is streamed, padded, and split into 512-token chunks.
2. **Continuous Action Space** – Instead of discrete `SKIP/UPDATE` actions, we predict a scalar $\lambda_t \in [0, n]$. This $\lambda_t$ scales the learning rate of the TTT update step. $\lambda_t \approx 0$ implies a "soft skip", while $\lambda_t \gg 1$ mimics multi-step updates.
3. **Gating Network** – A lightweight MLP observes 32-D features (loss, entropy, budget remaining) and outputs $\lambda_t$.
4. **End-to-End Training** – The entire system (Gating Net + TTT Adapter Parameters) is trained end-to-end.
   - **Objective**: $L_{total} = L_{main} + \beta L_{TTT} + \gamma \cdot \text{Cost}(\lambda_t)$.
   - We optimize the main task loss ($L_{main}$), the TTT reconstruction loss ($L_{TTT}$), and a budget penalty.
5. **Budget Awareness** – The gating network receives the "remaining budget" as a feature and is penalized for exceeding the target average cost.

## 3. Baselines
- **Fixed Schedules** – Always update with fixed intensity ($\\lambda=0, 1, 2, 4$).
- **RL (PPO)** – A discrete policy network trained with PPO to select update steps ($0, 1, 2, 4$). Used as a comparison point to demonstrate the efficiency of the differentiable approach.

## 4. Milestones
| Phase | Goals | Status |
|-------|-------|--------|
| Foundation | Pure NNX GPT-2, TTT layer, streaming pipeline | Complete |
| Differentiable Gating | Implement `GatingNetwork` and scaled TTT updates | Complete |
| Training Loop | End-to-end fine-tuning with budget constraints | Complete |
| Evaluation | Comparative study (Diff vs RL vs Fixed) | Complete |
| Scaling | Large-scale fine-tuning on TPU | Planned |

## 5. Experimental protocol
- **Training**: Fine-tune TTT parameters and Gating Network on The Stack v2 (Python). Base GPT-2 is frozen.
- **Metrics**: Validation Perplexity (Quality) vs. Average Compute Cost. We aim for a better Pareto frontier than fixed schedules.
- **Analysis**: Monitor the distribution of $\\lambda_t$ values. Does the model learn to "spend" updates on hard chunks and "save" on easy ones?

## 6. Risks & mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| Trivial Solutions | Medium | The model might collapse to always $\\lambda=0$ or $\\lambda=max$. Tune $\\gamma$ (cost weight) and $\\beta$ carefully. |
| Memory Usage | Medium | BPTT through TTT updates increases memory. Use gradient checkpointing or limit backprop horizon. |
| TPU Availability | High | Maintain CPU/GPU compatibility for development. |

## 7. Deliverables
1. **`train_differentiable.py`**: Main training script for adaptive TTT.
2. **`compare_methods.py`**: Script to compare Differentiable Gating against RL and Baselines.
3. **Trained Checkpoints**: Gating networks that demonstrate adaptive behavior.
4. **Analysis Reports**: Cost-Quality trade-off curves.
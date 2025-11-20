# PonderTTT Research Plan

## 1. Problem statement
We study adaptive test-time training for code generation models. A pretrained GPT-2 (125 M–1 B parameters) is frozen; a fast-weight adapter (TTT layer or LoRA) can be updated after each chunk of generated code. We must decide **when** and **how intensely** to update the adapter while respecting a strict compute budget. Unlike adaptive compute methods such as PonderNet, our actions mutate the model state; mistakes accumulate and future chunks become harder.

## 2. Method overview
1. **Chunked streaming** – The Stack v2 (Python) is streamed, padded with a dedicated `<|pad|>` token, and split into 512-token chunks. Each sequence yields multiple decision points.
2. **Action space** – `SKIP`, `UPDATE_1`, `UPDATE_2`, `UPDATE_4`, interpreted as 0/1/2/4 gradient-style updates on the fast-weight module. Costs are 1x, 3x, 6x, and 12x the baseline forward pass, matching the original TTT-LM accounting.
3. **Policy** – A PPO agent with a PID Lagrangian controller observes 32-D mask-aware features (loss delta, entropy, token stats, budget usage, history EMA). Rewards equal the chunk loss improvement minus λ·cost/limit; value loss is clipped and gradients are norm-clipped.
4. **Training loop** – For each rollout we reset the fast weights, stream `rollout_length` chunks, query the policy, execute updates, and log rewards/costs/KL. GAE runs with zero bootstrap for stability under truncation.
5. **Evaluation** – After training we run HumanEval/MBPP/ClassEval helpers; unsafe exec is gated by `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS=1` and should be enabled only in a sandbox. pass@k is reported with bootstrapped confidence intervals.

## 3. Baselines
- **SKIP** – no updates.
- **Fixed schedules** – UPDATE_1, UPDATE_2, UPDATE_4 on every chunk.
- **LoRA variants** – different ranks (64/128/256) to study memory/quality trade-offs.
- **Hand-crafted heuristics** – e.g., update when entropy exceeds a threshold. (Planned after PPO stabilizes.)

## 4. Milestones
| Phase | Goals | Status |
|-------|-------|--------|
| Foundation | Pure NNX GPT-2, TTT layer, streaming pipeline, scripts/tests | Complete |
| Chunk semantics | Implement chunk-level updates, real budgets, executable benchmarks | Complete |
| Policy stabilization | Tune PPO, add heuristic baselines, measure compute savings | In progress |
| Scaling | 350 M and 1 B models on TPU v4-64, statistical analysis, ablations | Planned |

## 5. Experimental protocol
- **Datasets** – The Stack v2 Python subset for training; HumanEval/MBPP/ClassEval for evaluation. `seq_length` must be divisible by `chunk_size`.
- **Metrics** – chunk loss/perplexity, average compute multiplier, pass@1/10/100 (depending on benchmark), KL drift, IQM and bootstrap CIs across seeds.
- **Seeds** – 3 for 125 M, 2 for 350 M (compute permitting). Policy training logs budgets, rewards, costs, and KL every iteration.
- **Hyperparameters** – `rollout_length` 64, `budget_limit` 3–5x, PPO clip 0.2, value clip 0.2, PPO epochs 4, entropy bonus 0.01, grad clip 1.0, PID gains (0.1, 0.01, 0.05). Adjust per scaling study.

## 6. Risks & mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| Policy stuck in SKIP/UPDATE_4 extremes | Medium | Curriculum on budget limits, entropy bonus annealing, heuristic warm-start |
| Dataset download latency | Medium | Pre-cache Hugging Face tokenizers, optionally stage Stack shards locally |
| Benchmark safety | Low | Exec gated by `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS`; integrate real sandbox for release |
| TPU availability | High | Maintain CPU/GPU-compatible configs so groundwork continues locally |

## 7. Deliverables
1. Reproducible training/evaluation scripts with documented configs.
2. Policy checkpoints showing strictly better cost-quality trade-offs than fixed schedules.
3. Analysis of action distributions, budget utilisation, and feature importance.
4. Technical report/paper summarizing the method and results.

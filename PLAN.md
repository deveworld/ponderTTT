# PonderTTT Research Plan

## 1. Problem statement
We study adaptive test-time training for code generation models. A pretrained GPT-2 (125 M–1 B parameters) is frozen; a fast-weight adapter (TTT layer or LoRA) can be updated after each chunk of generated code. We must decide **when** and **how intensely** to update the adapter while respecting a strict compute budget. Unlike adaptive compute methods such as PonderNet, our actions mutate the model state; mistakes accumulate and future chunks become harder.

## 2. Method overview
1. **Chunked streaming** – The Stack v2 (Python) is streamed, padded with a dedicated `<|pad|>` token, and split into 512-token chunks. Each sequence yields multiple decision points.
2. **Action space** – `SKIP`, `UPDATE_1`, `UPDATE_2`, `UPDATE_4`, interpreted as 0/1/2/4 gradient-style updates on the fast-weight module. Costs are 1×, 3×, 6×, and 12× the baseline forward pass, matching the original TTT-LM accounting.
3. **Policy** – A PPO agent with a PID Lagrangian controller observes 32-D features (loss delta, entropy, token statistics, budget usage, history EMA). Rewards equal the chunk loss improvement minus λ·cost/limit.
4. **Training loop** – For each rollout we reset the fast weights, stream `rollout_length` chunks, query the policy, execute updates, and log rewards/costs. GAE uses the critic’s bootstrap value to stabilize learning.
5. **Evaluation** – After training we run executable HumanEval/MBPP/ClassEval tests, reporting pass@k with bootstrapped confidence intervals.

## 3. Baselines
- **SKIP** – no updates.
- **Fixed schedules** – UPDATE_1, UPDATE_2, UPDATE_4 on every chunk.
- **LoRA variants** – different ranks (64/128/256) to study memory/quality trade-offs.
- **Hand-crafted heuristics** – e.g., update when entropy exceeds a threshold. (Planned after PPO stabilizes.)

## 4. Milestones
| Phase | Goals | Status |
|-------|-------|--------|
| Foundation | Pure NNX GPT-2, TTT layer, streaming pipeline, scripts/tests | ✅
| Chunk semantics | Implement chunk-level updates, real budgets, executable benchmarks | ✅
| Policy stabilization | Tune PPO, add heuristic baselines, measure compute savings | ⏳
| Scaling | 350 M and 1 B models on TPU v4-64, statistical analysis, ablations | ⏳

## 5. Experimental protocol
- **Datasets** – The Stack v2 Python subset for training; HumanEval/MBPP/ClassEval for evaluation.
- **Metrics** – chunk loss/perplexity, average compute multiplier, pass@1/10/100 (depending on benchmark), IQM and bootstrap CIs across seeds.
- **Seeds** – 3 for 125 M, 2 for 350 M (compute permitting). Policy training logs budgets and rewards every iteration.
- **Hyperparameters** – `rollout_length` 64, `budget_limit` 3–5×, PPO clip 0.2, entropy bonus 0.01, PID gains (0.1, 0.01, 0.05). Adjust per scaling study.

## 6. Risks & mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| Policy stuck in SKIP/UPDATE_4 extremes | Medium | Curriculum on budget limits, entropy bonus annealing, heuristic warm-start |
| Dataset download latency | Medium | Pre-cache Hugging Face tokenizers, optionally stage Stack shards locally |
| Benchmark safety | Low | Executions run in a tightly-scoped namespace; consider `sandboxed_eval` for release |
| TPU availability | High | Maintain CPU/GPU-compatible configs so groundwork continues locally |

## 7. Deliverables
1. Reproducible training/evaluation scripts with documented configs.
2. Policy checkpoints showing strictly better cost-quality trade-offs than fixed schedules.
3. Analysis of action distributions, budget utilisation, and feature importance.
4. Technical report/paper summarizing the method and results.

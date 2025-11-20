# Project Status

**Phase**: Chunk-Level Infrastructure

| Area | Status | Notes |
|------|--------|-------|
| NNX GPT-2 + TTT/LoRA fast weights | Complete | Base model frozen via `stop_gradient`, weight tying verified |
| Streaming data pipeline | Complete | `<|pad|>` token added, chunk + mask tensors propagated everywhere |
| Chunk semantics | Complete | Baseline & policy loops issue true SKIP/UPDATE_k actions |
| PPO + PID controller | Complete | Cost-penalized rewards, clipped value loss, multi-epoch PPO, grad clipping, KL logging |
| Executable benchmarks | Complete | HumanEval/MBPP/ClassEval helpers gated by `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS` for sandboxed exec |
| Tooling/tests | Complete | `scripts/quick_test.py`, `test_pipeline.py`, distributed and TPU setup scripts updated; checkpoint errors surfaced |
| Large-scale experiments | In progress | Requires TPU v4-64 or multi-GPU cluster |

## Recent work
- Implemented chunk-level training utilities shared by baselines and the policy trainer.
- Rebuilt policy rollouts to track real budgets and store the full batch for PPO.
- Fixed the tokenizer/padding bug that masked genuine `<|endoftext|>` tokens.
- Replaced placeholder benchmark stubs with executable pass@k evaluation.
- Refreshed every helper script to match the NNX stack.

## Next steps
1. **Stabilize PPO** – sweep PID gains, rollout length, entropy bonus, KL targets; record cost-quality curves.
2. **Add heuristics** – simple entropy or loss-threshold policies to compare against PPO.
3. **Scaling** – run 350 M and 1 B experiments on TPU v4-64 once hardware is available; enable gradient checkpointing and parameter sharding.
4. **Evaluation package** – integrate a real sandbox (e.g., `evalplus`/firejail) so `evaluate_all` can run without `PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS`.

## How to validate changes locally
```bash
python scripts/quick_test.py
python scripts/test_pipeline.py
python scripts/test_distributed.py --multi_host   # optional
python scripts/test_weight_tying.py
pytest tests/test_checkpointing.py                # Orbax save/load regression
```

## Outstanding risks
- **Hardware availability** – TPU access dictates progress on large models; keep smaller configs and synthetic benchmarks for debugging.
- **Policy collapse** – continue monitoring action entropy; curriculum or KL penalties ready if needed.
- **Dataset download** – streaming relies on Software Heritage; consider mirroring shard subsets for offline work.

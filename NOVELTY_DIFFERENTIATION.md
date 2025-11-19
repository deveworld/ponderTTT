# Novelty & Differentiation

## Core idea
PonderTTT is the first framework that learns **when to update fast weights during test-time adaptation** for code generation models. Every action mutates the model state; the policy must budget irreversible updates across a chunk stream. This differs from halting/routing work (PonderNet, CALM, MoE) where parameters stay fixed, and from classic TTT (TENT, MEMO, LaCT) where every chunk receives the same update schedule.

## Why gradients are insufficient
- Fast-weight updates make the feature distribution non-stationary. The policy’s inputs change after every action; simple halting gradients (PonderNet style) break.
- Actions are irreversible. A bad UPDATE_4 corrupts all future chunks. Exploration must reason about long-term cost, so we wrap PPO with a PID Lagrangian controller to enforce the compute budget.
- Credit assignment spans many chunks. Improvements several chunks later must be attributed to earlier updates, which is why we keep full chunk traces, compute GAE, and expose history features.

## Architectural choices
| Component | Prior art | PonderTTT difference |
|-----------|-----------|----------------------|
| Base model | TTT-LM (JAX) | Pure Flax NNX rewrite with chunk-level hooks, tied embeddings, and optional LoRA |
| Fast weights | TTT Layer / LoRA | Both available; UPDATE_k literally runs k optimizer steps on the fast weights |
| Policy | – | Actor-critic with 32-D signals: entropy/perplexity, token diversity, EMA of difficulty/cost, true budget remaining |
| Evaluation | pass@k templates | Executable HumanEval/MBPP/ClassEval directly from the CLI |

## Practical benefits
1. **Budget-aware compute allocation** – 30–40% compute savings are achievable once the policy learns to skip easy boilerplate chunks and invest updates in harder ones.
2. **Transparent accounting** – Each training run logs loss, perplexity, action frequencies, and cost multipliers per chunk; visualization scripts make budget usage obvious.
3. **Research platform** – Everything is TPU-ready, uses JAX meshes and sharding primitives, and exposes clean hooks for ablation studies.

## FAQ
**Is this just PonderNet with different wording?** No. PonderNet decides how long to keep thinking with fixed parameters. PonderTTT decides when to mutate fast weights; this is a constrained RL problem rather than a differentiable halting problem.

**Why NNX instead of Transformers?** NNX keeps state in Python objects (easier checkpoints), integrates with new JAX sharding APIs, and removes the runtime dependency on `transformers`. We still provide a converter if you want to import Hugging Face weights.

**Can I swap in LoRA or another adapter?** Yes. Set `--fast_weight_type lora` and provide a `LoRAConfig`. The action semantics remain the same; UPDATE_k applies k optimizer steps to the LoRA parameters instead of the TTT layer.

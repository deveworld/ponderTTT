# PonderTTT Research Plan

## Current Status: arXiv Preprint Complete

The initial version of PonderTTT has been validated on GPT-2 (125M, 350M) with promising results:
- 4.5x perplexity improvement on held-out Python code
- Strong OOD generalization to JavaScript (2.5x), Java (6.2x)
- Binary gating via Gumbel-Softmax enables true computational savings

---

## Phase 2: Conference Submission Roadmap

### 1. Scale to Gemma 3 (4B + 12B)
**Goal**: Validate PonderTTT on modern, production-relevant architectures at two scales.

| Task | Details | Priority |
|------|---------|----------|
| Gemma 3 4B Integration | Frozen backbone, ~8GB VRAM (bf16) | P0 |
| Gemma 3 12B Integration | Frozen backbone, ~24GB VRAM (bf16) | P0 |
| TTT Layer Adaptation | Adjust hidden dims for Gemma 3 architecture | P0 |
| Memory Optimization | Gradient checkpointing, mixed precision (bf16) | P0 |
| Infrastructure | Multi-GPU for 12B, efficient data loading | P1 |

**Model Specs**:
| Model | Parameters | Memory (bf16) | Memory (int4) | Context |
|-------|------------|---------------|---------------|---------|
| Gemma 3 4B | 4B | 8 GB | 2.6 GB | 128K |
| Gemma 3 12B | 12B | 24 GB | 6.6 GB | 128K |

**Checkpoints**:
- [ ] Gemma 3 4B baseline on Python (The Stack v2)
- [ ] Gemma 3 12B baseline on Python
- [ ] PonderTTT integration with both models
- [ ] Training stability verification (10K steps each)
- [ ] Scaling analysis: 4B vs 12B improvement comparison

### 2. LoRA-TTT for Efficiency
**Goal**: Replace full TTT updates with Low-Rank Adaptation for practical wall-clock speedups.

| Task | Details | Priority |
|------|---------|----------|
| LoRA Layer Implementation | Rank-16/32 adaptation matrices for TTT | P0 |
| Wall-clock Benchmarking | Measure actual latency, not just theoretical FLOPs | P0 |
| Comparative Analysis | Full TTT vs LoRA-TTT (quality vs speed tradeoff) | P1 |
| Memory Profiling | Peak memory usage comparison | P1 |

**Target Metrics**:
- Wall-clock latency <= 1.3x baseline (vs current 1.74x)
- Quality degradation < 5% compared to full TTT

### 3. Reasoning Benchmark Evaluation
**Goal**: Demonstrate effectiveness beyond perplexity on downstream reasoning tasks.

| Benchmark | Domain | Metric | Priority |
|-----------|--------|--------|----------|
| **MATH500** | Mathematical reasoning | Accuracy (pass@1) | P0 |
| **GSM8K** | Grade-school math | Accuracy | P0 |
| **LiveCodeBench** | Code generation (recent) | Pass rate | P0 |
| **GPQA-Diamond** | Science QA (graduate-level) | Accuracy | P1 |

**Evaluation Protocol**:
- Few-shot prompting (0-shot, 5-shot)
- Compare: Base -> Base+TTT(fixed) -> Base+PonderTTT
- Report improvement over non-adaptive TTT baseline

### 4. Advanced Gating Features
**Goal**: Enrich gating decisions with uncertainty-aware signals beyond hidden states.

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **Prediction Entropy** | H(p) = -sum(p_i * log(p_i)) of output distribution | Compute from logits |
| **Variance of Gradients (VOG)** | Gradient magnitude variability across tokens | Track during forward pass |
| **Attention Dispersion** | Entropy of attention weights (scattered = uncertain) | Extract from attention layers |
| **Token-level Confidence** | Max probability of predicted token | Simple softmax max |

**Updated Gating Network**:
```
Input Features (expanded):
- Hidden state statistics (mean, std) [current]
- Prediction entropy [new]
- Attention entropy per head [new]
- VOG estimate [new]
- Remaining budget [current]

Architecture:
- Feature dimension: 32 -> 64
- Add feature normalization per signal type
```

---

## Updated Milestones

| Phase | Goals | Status |
|-------|-------|--------|
| Foundation | Pure NNX GPT-2, TTT layer, streaming pipeline | Complete |
| Differentiable Gating | Binary gating via Gumbel-Softmax | Complete |
| Training Loop | End-to-end fine-tuning with budget constraints | Complete |
| arXiv Submission | Initial paper with GPT-2 results | Complete |
| Gemma 3 Integration | Gemma 3 4B + 12B backbone | Planned |
| LoRA-TTT | Efficient low-rank TTT updates | Planned |
| Reasoning Benchmarks | MATH500, GSM8K, LiveCodeBench, GPQA | Planned |
| Advanced Gating | Entropy, VOG, attention-based features | Planned |
| Conference Submission | Full paper with Gemma 3 results | Target |

---

## Risk Assessment (Phase 2)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gemma 3 12B Memory (24GB) | High | LoRA-TTT, gradient checkpointing, bf16/int4 |
| TTT at Scale Instability | Medium | Careful LR scheduling, warmup |
| Benchmark Variance | Medium | Multiple seeds, statistical significance tests |
| Compute Requirements | High | Start with 4B, scale to 12B after validation |

---

## Deliverables (Phase 2)

1. **`models/gemma3_nnx.py`**: Gemma 3 (4B, 12B) implementation in NNX
2. **`models/lora_ttt_layer.py`**: LoRA-based efficient TTT layer
3. **`evaluation/reasoning_benchmarks.py`**: MATH500, GSM8K, LiveCodeBench, GPQA evaluation
4. **`models/advanced_gating.py`**: Entropy/VOG/attention-aware gating network
5. **Conference Paper**: Full results on Gemma 3 (4B + 12B) with reasoning benchmarks

---

## Phase 1 Reference (Completed)

### Problem Statement
Adaptive test-time training for code generation models. A pretrained GPT-2 (125M, 350M) is frozen; a fast-weight adapter (TTT layer) is updated after each chunk. We decide **when** to update via learned binary gating.

### Method Overview
1. **Chunked streaming** - The Stack v2 (Python) split into 512-token chunks
2. **Binary Gating** - SKIP/UPDATE decisions via Gumbel-Softmax
3. **Gating Network** - Lightweight MLP (6,466 params) observes hidden state features
4. **End-to-End Training** - L_total = L_CE + 0.1 * L_TTT + 0.1 * L_cost
5. **Budget Awareness** - Cost penalty enforces computational budget

### Key Results
| Model | Baseline PPL | PonderTTT PPL | Improvement | Cost |
|-------|-------------|---------------|-------------|------|
| GPT-2 125M | 26.36 | 5.85 | 4.5x | 2.67x |
| GPT-2 350M | 26.13 | 5.99 | 4.4x | 2.67x |

**OOD Generalization (125M, trained on Python):**
- JavaScript: 15.29 → 6.01 (2.5x)
- Java: 42.18 → 6.85 (6.2x)
- Go: 1004 → 14.27 (70x)

### Baselines
- **Fixed Schedules** - SKIP (0 updates), UPDATE_1, UPDATE_2, UPDATE_4

### Phase 1 Deliverables (Complete)
1. `train_hard_skip.py`: Main training script for binary gating (Gumbel-Softmax)
2. `train_differentiable.py`: Alternative continuous gating script (soft skip)
3. `compare_methods.py`: Comparison against baselines
4. Trained Checkpoints: Gating networks with adaptive behavior
5. arXiv Paper: "Learning to Ponder: Adaptive Compute Allocation via Test-Time Training"

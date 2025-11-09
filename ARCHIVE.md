# PonderTTT: 방법론 아카이브

**작성일**: 2025-11-09
**목적**: 코드 재작성 전 핵심 아이디어 및 방법론 보존
**주의**: 이 문서는 순수하게 아이디어와 방법론만 기록합니다.

---

## Executive Summary

PonderTTT는 Test-Time Training (TTT) 아키텍처에 **적응형 계산 할당**을 적용하는 연구입니다. 핵심 아이디어는 각 토큰의 **난이도에 따라 gradient descent 반복 횟수를 동적으로 조정**하여 계산 효율성을 높이는 것입니다.

**핵심 질문**: "모든 토큰이 동일한 계산량을 필요로 하는가?"

**가설**: 난이도가 높은 토큰은 더 많은 반복이 필요하고, 쉬운 토큰은 적은 반복으로 충분하다.

**접근법**: 강화학습(REINFORCE)을 사용하여 토큰별 최적 반복 횟수를 학습한다.

**⚠️ 주요 우려사항**:
1. 🔴 **Policy overhead**: BiLSTM (14.2M params) 비용이 절감분 상쇄 가능 → Net gain 불확실
2. 🔴 **비현실적 기대**: 15-30% 절감은 과도함 → 0-5% 또는 negative 예상
3. ⚠️ **WikiText-2 너무 작음**: 2M tokens → WikiText-103 (100M) 필요
4. ⚠️ **REINFORCE 불안정**: High variance → Training 어려움
5. ⚠️ **Null hypothesis 가능성**: 가설 자체가 틀릴 수 있음 → Negative result도 학술적 기여

**현실적 목표**: Interpretability 확보 (성공), Efficiency gain은 보너스 (불확실)

---

## 1. 연구 동기 (Motivation)

### 1.1 배경: Test-Time Training (TTT)

**TTT의 핵심 개념**:
- Hidden state를 학습 가능한 모델로 취급
- 각 토큰 처리 시 self-supervised learning으로 모델 업데이트
- 업데이트된 모델이 다음 토큰의 hidden state가 됨

**TTT의 이점**:
- 긴 문맥 처리 능력
- 선형 복잡도 (Self-attention은 quadratic)
- Test-time adaptation

**TTT의 한계**:
- 모든 토큰에 대해 동일한 계산량 사용
- 토큰 난이도를 고려하지 않음
- 계산 자원의 비효율적 사용 가능성

### 1.2 동기: 적응형 계산 할당

**관찰**:
- 모든 토큰이 동일한 난이도를 가지지 않음
- 쉬운 토큰 (예: "the", "a"): 적은 계산으로 충분
- 어려운 토큰 (예: 전문용어, 맥락 의존적 단어): 더 많은 계산 필요

**아이디어**:
- 토큰별로 gradient descent 반복 횟수 K를 동적으로 할당
- 난이도 높은 토큰 → 큰 K (예: K=8)
- 난이도 낮은 토큰 → 작은 K (예: K=1)

**기대 효과**:
- 전체 계산량은 유지하면서 성능 향상, 또는
- 성능은 유지하면서 계산량 절감

---

## 2. 핵심 방법론

### 2.1 문제 정의

**입력**: 토큰 시퀀스 X = [x₁, x₂, ..., xₙ]

**목표**: 각 토큰 xₜ에 대해 최적 반복 횟수 Kₜ ∈ {1, 2, 4, 8} 결정

**제약조건**:
- 평균 반복 횟수가 목표 값 (예: K_avg = 4) 유지
- 전체 FLOPs 예산 초과 금지

**최적화 목표**:
```
Maximize: 언어 모델 성능 (Minimize perplexity)
Subject to: E[Kₜ] ≤ K_target
```

### 2.2 적응형 계산 할당 프레임워크

#### 2.2.1 Iterative Gradient Descent

**기본 아이디어**:
- 각 토큰에 대해 fast-weight 모델 W를 Kₜ번 업데이트
- Self-supervised loss: L = ||W @ K - (V - K)||²

**업데이트 규칙**:
```
For token t:
  W₀ = W from previous token
  For k = 1 to Kₜ:
    Compute loss: L = ||W_{k-1} @ K_t - (V_t - K_t)||²
    Compute gradient: ∇_W L
    Update: W_k = W_{k-1} - η * ∇_W L

  Use W_{Kₜ} for output computation
```

**Sequential carry-over**:
- Token t의 최종 weight W_{Kₜ}가 token t+1의 초기 weight W₀가 됨
- 이전 토큰의 학습이 다음 토큰에 영향

#### 2.2.2 학습된 Halting Policy

**Policy Network 구조**:
- 입력: 토큰의 hidden state representation
- 출력: K ∈ {1, 2, 4, 8}에 대한 확률 분포

**Architecture**:
1. **Context Encoder**: Bidirectional LSTM으로 local context 포착
2. **Step Predictor**: 2-layer MLP로 K 예측

**Policy output**:
```
π_θ(K_t | h_t) = Categorical distribution over {1, 2, 4, 8}
```

### 2.3 강화학습: REINFORCE Algorithm

#### 2.3.1 기본 REINFORCE

**Objective**:
```
J(θ) = E_{K ~ π_θ}[R(K)]

where R(K) = -Loss(K)  (reward is negative loss)
```

**Policy Gradient**:
```
∇_θ J(θ) = E[∇_θ log π_θ(K_t | h_t) * (R_t - b_t)]

where:
- R_t: reward at position t
- b_t: baseline (moving average of rewards)
```

**학습 알고리즘**:
1. Policy π_θ로 K 샘플링 (stochastic during training)
2. 샘플링된 K로 forward pass 수행
3. Per-token loss 계산
4. Policy gradient로 θ 업데이트

#### 2.3.2 Temporal Credit Assignment

**핵심 아이디어**: Token t의 K 결정이 **미래 토큰들의 성능에도 영향**

**이유**:
- Fast-weight가 sequential하게 carry-over됨
- Token t에서 큰 K 사용 → 더 나은 W 학습
- 더 나은 W → Token t+1, t+2, ...의 성능 향상

**Monte Carlo Returns**:
```
G_t = Σ_{i=t}^{T} γ^{i-t} * r_i

where:
- r_i: reward at position i (negative loss)
- γ: discount factor (예: 0.99)
- G_t: discounted cumulative return
```

**Policy Gradient with Temporal Credit**:
```
∇_θ J(θ) = E[∇_θ log π_θ(K_t | h_t) * (G_t - b_t)]
```

**Interpretation**:
- γ=0: per-token reward만 고려 (myopic)
- γ=0.99: 미래 100 토큰까지 영향 고려 (far-sighted)

#### 2.3.3 Compute Budget Constraint

**문제**: Policy가 항상 K=8을 선택할 수 있음

**해결**: Regularization term 추가

**Objective with constraint**:
```
J(θ) = E[R(K)] - λ * |E[K] - K_target|

where:
- λ: compute penalty coefficient (예: 0.01)
- K_target: 목표 평균 K (예: 4.0)
```

**Interpretation**:
- E[K] > K_target: penalty 부과 (너무 많은 계산 사용)
- E[K] < K_target: penalty 없음 (효율적)

### 2.4 난이도 측정 (Heuristic Baselines)

학습된 policy와 비교하기 위한 heuristic 기반 방법들:

#### 2.4.1 Entropy-based Policy

**아이디어**: 예측 불확실성이 높은 토큰이 어려움

**난이도 측정**:
```
difficulty(x_t) = H(p(x_{t+1} | x_{≤t}))

where H(p) = -Σ p(x) log p(x)
```

**K 할당**:
- High entropy → K=8
- Medium entropy → K=4 or K=2
- Low entropy → K=1

#### 2.4.2 Loss-based Policy

**아이디어**: 현재 모델이 예측하기 어려운 토큰이 어려움

**난이도 측정**:
```
difficulty(x_t) = -log p(x_t | x_{<t})  (cross-entropy loss)
```

**K 할당**: Loss 값에 따라 threshold-based assignment

#### 2.4.3 Gradient-based Policy

**아이디어**: Gradient magnitude가 큰 토큰이 어려움

**난이도 측정**:
```
difficulty(x_t) = ||∇_W L(x_t)||
```

#### 2.4.4 Calibration

**문제**: Heuristic 값의 범위가 데이터에 따라 다름

**해결**: Validation set에서 percentile-based threshold 계산

**알고리즘**:
1. Validation set에서 모든 토큰의 difficulty 계산
2. Percentile 계산 (예: 25%, 50%, 75%)
3. Threshold 설정:
   - difficulty < p25 → K=1
   - p25 ≤ difficulty < p50 → K=2
   - p50 ≤ difficulty < p75 → K=4
   - difficulty ≥ p75 → K=8

---

## 3. 실험 설계

### 3.1 비교 방법론

**Baseline methods**:
1. **Uniform K=1**: 모든 토큰에 K=1 (minimal compute)
2. **Uniform K=2**: 모든 토큰에 K=2
3. **Uniform K=4**: 모든 토큰에 K=4 (reference)
4. **Uniform K=8**: 모든 토큰에 K=8 (maximum compute)

**Heuristic methods**:
5. **Entropy-based**: Entropy로 K 결정
6. **Loss-based**: Loss로 K 결정
7. **Gradient-based**: Gradient norm으로 K 결정

**Learned methods**:
8. **Learned (λ=0.01, target=4)**: REINFORCE with compute constraint
9. **Learned (λ=0.05, target=4)**: Stronger compute penalty
10. **Learned (λ=0.01, no target)**: No compute constraint

### 3.2 평가 지표

**Primary metrics**:
- **Perplexity**: 언어 모델 성능 (lower is better)
- **FLOPs**: 계산 비용 (lower is better)

**Pareto frontier**:
- X축: FLOPs
- Y축: Perplexity
- 목표: Pareto optimal 달성

**Statistical tests**:
- Multiple seeds (10+)
- Paired t-test
- Bonferroni correction
- Confidence intervals

### 3.3 성공 기준

**Hypothesis**: 학습된 adaptive policy가 uniform baseline보다 우수

**성공 조건** (보수적 설정):
1. **Quality maintenance**: Perplexity가 Uniform-K4 대비 1% 이내
2. **Efficiency gain**: Net FLOPs (policy overhead 포함) 가 5% 이상 절감
3. **Statistical significance**: p < 0.05 (Bonferroni corrected)
4. **Difficulty correlation**: Optimal K와 difficulty metric 간 r > 0.3

**⚠️ 주의**: Policy overhead로 인해 net gain이 0%일 가능성도 고려해야 함

### 3.4 Oracle Analysis

**목적**: Adaptive allocation의 upper bound 추정

**방법**:
1. 각 토큰에 대해 K ∈ {1,2,4,8} 모두 시도
2. Per-token loss 측정
3. 각 토큰의 optimal K 선택

**분석**:
- Oracle K distribution
- Oracle performance (best possible)
- Learned policy가 oracle에 얼마나 근접하는가?

---

## 4. 주요 연구 질문

### RQ1: Adaptive allocation이 효과적인가?

**질문**: "동일한 계산 예산으로 adaptive가 uniform보다 나은가?"

**실험**:
- Uniform K=4 vs Learned (target=4)
- FLOPs 매칭 시 perplexity 비교

**Expected**: Learned < Uniform (lower perplexity)

### RQ2: 어떤 난이도 metric이 가장 좋은가?

**질문**: "Entropy, loss, gradient 중 무엇이 optimal K와 가장 상관관계가 높은가?"

**실험**:
- Oracle K 계산
- 각 metric과의 correlation 측정

**Expected**: Loss-based metric이 가장 높은 correlation

### RQ3: Temporal credit assignment가 도움이 되는가?

**질문**: "γ=0.99 (temporal) vs γ=0.0 (per-token) 중 무엇이 나은가?"

**실험**:
- Learned with γ=0.99
- Learned with γ=0.0
- Performance 비교

**Expected**: γ=0.99가 더 나음 (sequential dependency 고려)

### RQ4: Compute constraint의 영향은?

**질문**: "λ 값이 quality-efficiency tradeoff에 어떤 영향을 주는가?"

**실험**:
- λ ∈ {0.0, 0.01, 0.05, 0.1}
- Pareto curve 그리기

**Expected**: λ가 클수록 낮은 FLOPs, 약간 높은 perplexity

---

## 5. 관련 연구와의 관계

### 5.1 Adaptive Computation Time (ACT)

**ACT (Graves 2016)**:
- RNN에서 각 토큰의 **thinking time** 동적 조정
- Halting probability 학습

**PonderTTT vs ACT**:
- 유사: 둘 다 adaptive compute allocation
- 차이:
  - ACT: Layer depth 조정 (동일 layer를 여러 번)
  - PonderTTT: Test-time gradient descent 반복 횟수 조정

### 5.2 Mixture of Depths (MoD)

**MoD (Raposo et al., 2024)**:
- 일부 토큰만 full transformer layer 통과
- 나머지는 skip connection

**PonderTTT vs MoD**:
- 유사: 토큰별 compute 차별화
- 차이:
  - MoD: Layer 전체를 skip할지 결정
  - PonderTTT: TTT 내부의 반복 횟수 조정

### 5.3 SIFT (Efficient Data Selection)

**SIFT (Mindermann et al., 2022)**:
- Training 중 어려운 data에 집중
- Data weighting

**PonderTTT vs SIFT**:
- 유사: Difficulty-aware resource allocation
- 차이:
  - SIFT: Data selection (training)
  - PonderTTT: Per-token compute (inference)

### 5.4 Official TTT

**Official TTT (Sun et al., 2024)**:
- Test-time training with self-supervised learning
- Linear complexity

**PonderTTT vs Official TTT**:
- 유사: 둘 다 TTT 사용
- 차이:
  - Official: 모든 토큰 동일 처리
  - PonderTTT: 토큰별 adaptive compute

**PonderTTT의 기여**: Official TTT 위에 adaptive compute 레이어 추가

### 5.5 차별점 요약

| Method | Allocation Target | Our Contribution |
|--------|------------------|------------------|
| ACT | Layer depth | TTT iteration count |
| MoD | Layer skip/compute | Fine-grained iteration control |
| SIFT | Data selection | Per-token compute |
| Official TTT | Fixed compute | Adaptive allocation |

---

## 6. 기대 기여

### 6.1 학술적 기여

**Contribution 1: Adaptive TTT Framework**
- TTT에 처음으로 adaptive compute 적용
- REINFORCE로 optimal allocation 학습

**Contribution 2: Temporal Credit Assignment**
- Sequential dependency를 고려한 policy learning
- Monte Carlo returns with discount factor

**Contribution 3: Comprehensive Evaluation**
- Multiple baselines (uniform, heuristic, learned)
- Oracle analysis for upper bound
- Statistical validation

### 6.2 실용적 가치 (현실적 평가)

**⚠️ 중요**: 아래 기대치는 낙관적이며, null result 가능성도 고려해야 함

**Efficiency gain (불확실)**:
- 이상적 시나리오: 5-10% net FLOPs 절감
- 현실적 시나리오: Policy overhead가 절감분 상쇄 (0% gain)
- 최악 시나리오: Net negative (policy 비용 > 절감분)
- **Policy overhead**: BiLSTM (14.2M params) 실행 비용 무시 불가

**Quality improvement (불확실)**:
- 이상적 시나리오: Marginal improvement (< 1% perplexity)
- 현실적 시나리오: Uniform K=4와 통계적으로 차이 없음
- **Null hypothesis 가능성**: 모든 토큰이 실제로 비슷한 계산 필요할 수 있음

**Interpretability (확실)**:
- 어떤 토큰이 어려운지 policy가 학습 (성공 여부와 무관)
- Model debugging 및 분석에 도움
- 이것만으로도 연구 가치 있음

---

## 7. 한계 및 향후 연구

### 7.1 현재 한계

**Dataset scale** (⚠️ Major):
- WikiText-2는 너무 작음 (2M tokens only)
- Policy 학습에 충분한 데이터인가 불확실
- WikiText-103 (100M tokens) 최소 필요

**Model size**:
- 60M parameters는 작은 편
- Larger model에서 효과 검증 필요

**Domain**:
- Language modeling only
- Other domains (vision, speech) 검증 필요

**Policy overhead** (🔴 Critical):
- HaltingPolicyNetwork: 14.2M parameters (전체 모델의 23.6%)
- 매 토큰마다 BiLSTM forward pass 필요
- Overhead가 절감분을 상쇄할 가능성 높음
- **Net gain이 negative일 수 있음**

### 7.2 주요 우려사항 및 리스크

**🔴 Critical Risks** (프로젝트 실패 가능성):

**1. Policy overhead > Savings**
- **문제**: BiLSTM 실행 비용이 K 절감으로 얻는 이득보다 클 수 있음
- **분석**:
  - Policy forward: ~14M params × 토큰당
  - TTT iteration savings: K 감소분 × TTT params
  - Net gain = Savings - Overhead (음수 가능)
- **완화**:
  - Lightweight policy (MLP-only, no LSTM)
  - Amortized policy (mini-batch level)
  - Oracle 분석으로 upper bound 먼저 확인

**2. 비현실적 기대치**
- **문제**: 15-30% 절감은 과도하게 낙관적
- **현실**:
  - Policy overhead 고려 시 0-5% 또는 negative
  - Null result 가능성 높음
- **대응**:
  - 보수적 목표 설정 (5% net gain)
  - Negative result도 학술적 기여로 인정

**⚠️ Major Concerns** (연구 난이도):

**3. WikiText-2 너무 작음**
- **문제**: 2M tokens로 policy 학습 어려움
- **증상**: Overfitting, 불안정한 training
- **해결**: WikiText-103 (100M tokens) 필수

**4. REINFORCE 불안정**
- **문제**: High variance gradients
- **증상**:
  - Training이 수렴하지 않음
  - Policy가 degenerate solution 학습 (항상 K=1 또는 K=8)
  - Reward signal 너무 sparse
- **완화**:
  - Strong baseline (value network)
  - Entropy regularization
  - Curriculum learning (easy → hard)
  - PPO 고려

**5. Null Hypothesis 가능성**
- **문제**: 핵심 가설이 틀릴 수 있음
- **가설**: "토큰별로 다른 계산 필요"
- **반례**: 실제로 모든 토큰이 비슷한 K 필요할 수 있음
- **Oracle 분석 결과가 uniform distribution이면?**
  - 이것도 중요한 negative result
  - "Adaptive allocation은 효과 없다" 증명
  - 학술적 기여 여전히 존재

**연구 성공/실패 시나리오**:

| 시나리오 | Net Gain | 학술적 가치 | 실용적 가치 |
|---------|----------|------------|------------|
| Best case | +5~10% | 높음 | 높음 |
| Good case | +1~5% | 높음 | 중간 |
| Null result | 0% | 중간 | 낮음 |
| Negative | < 0% | 낮음 | 없음 |

**Null result 대비 전략**:
- Oracle analysis를 먼저 수행 (upper bound 확인)
- Heuristic baselines로 feasibility 검증
- Interpretability에 집중 (efficiency 못 얻어도 분석 도구로 가치)

### 7.3 향후 연구 방향

**Direction 1: Scaling**
- WikiText-103, C4, The Pile
- 1B+ parameter models
- Multi-domain evaluation

**Direction 2: Advanced Policies**
- Transformer-based policy
- Multi-granularity (token + layer)
- Learned adaptive mini-batch size

**Direction 3: Theoretical Analysis**
- Convergence guarantees
- Sample complexity bounds
- Optimal allocation theory

**Direction 4: Hybrid Approaches**
- Combine heuristic + learned
- Multi-stage allocation
- Dynamic threshold adaptation

**Direction 5: Other Architectures**
- Adaptive compute for Mamba
- Adaptive compute for RWKV
- Generalization to linear RNNs

---

## 8. 핵심 인사이트

### 8.1 Why Adaptive Compute for TTT?

**Observation**: TTT updates fast-weight at every token

**Problem**: Not all tokens need same amount of updating

**Insight**:
- Easy tokens: W is already good, 1-2 steps enough
- Hard tokens: W needs more refinement, 4-8 steps needed

**Benefit**: Allocate compute where it matters most

### 8.2 Why REINFORCE?

**Alternative 1**: Fixed heuristics (entropy, loss)
- Pro: Simple, no training
- Con: Not optimal, no adaptation

**Alternative 2**: Differentiable gating (Gumbel-softmax)
- Pro: End-to-end differentiable
- Con: Biased gradients, complex

**REINFORCE choice**:
- Pro: Unbiased gradients, flexible
- Pro: Natural for discrete actions (K ∈ {1,2,4,8})
- Con: High variance (mitigate with baseline)
- ⚠️ **주의**: Training 불안정성 예상, PPO 고려 필요

### 8.3 Why Temporal Credit?

**Key insight**: Sequential dependency in TTT

**Mechanism**:
- Token t with K=8 → Better W learned
- Better W → Token t+1 benefits (even with K=1)

**Implication**: Policy must consider long-term effects

**Solution**: Monte Carlo returns with γ=0.99

---

## 9. 방법론적 선택의 근거

### 9.1 Discrete K ∈ {1, 2, 4, 8}

**Why discrete?**
- Hardware efficiency (powers of 2)
- Easier to profile and optimize
- Clear semantic meaning

**Why these values?**
- K=1: Minimal compute baseline
- K=2, K=4: Intermediate options
- K=8: Maximum reasonable compute
- Covers wide range (1× to 8×)

### 9.2 Per-token (not per-layer)

**Why per-token?**
- TTT processes tokens sequentially
- Each token has different difficulty
- Finer granularity than per-layer

**Alternative**: Per-mini-batch
- Official TTT uses mini-batch
- PonderTTT uses per-token for adaptivity

### 9.3 REINFORCE (not other RL)

**Why not Q-learning?**
- Discrete action space (4 options)
- Policy gradient more natural

**Why not PPO?**
- Overkill for this problem
- REINFORCE + baseline sufficient

**Why not A3C?**
- Single sequence, not parallel envs
- No need for actor-critic

---

## 10. 평가 방법론 설계

### 10.1 Fairness in Comparison

**Challenge**: Different methods use different compute

**Solution 1**: Match average K
- Uniform K=4 vs Learned (target=4)
- Same expected FLOPs

**Solution 2**: Pareto frontier
- Plot all methods on efficiency-quality space
- Find dominating solutions

**Metrics**:
- Perplexity (quality)
- FLOPs (efficiency)
- Pareto optimality

### 10.2 Statistical Rigor

**Multiple seeds**: 10+ for variance estimation

**Paired tests**: Same data, different methods

**Corrections**: Bonferroni for multiple comparisons

**Effect size**: Cohen's d, not just p-values

**Bootstrap CI**: Non-parametric confidence intervals

### 10.3 Ablation Studies

**Essential ablations**:
1. γ ablation: {0.0, 0.5, 0.9, 0.99}
2. λ ablation: {0.0, 0.01, 0.05, 0.1}
3. Policy architecture: LSTM vs MLP
4. Step options: {1,2,4,8} vs {1,4} vs {1,2,4,8,16}

---

## 11. 재구현 시 고려사항

### 11.1 핵심 결정 사항

**Decision 1**: Iterative vs Analytical TTT?
- Current: Iterative (K-step GD)
- Alternative: Analytical with adaptive mini-batch size
- Tradeoff: Flexibility vs theoretical guarantees

**Decision 2**: Policy granularity?
- Current: Per-token
- Alternative: Per-position (averaged over batch)
- Tradeoff: Adaptivity vs stability

**Decision 3**: Discount factor γ?
- Current: 0.99 (fixed)
- Alternative: Learned, position-dependent
- Requires: Ablation to validate choice

**Decision 4**: Step options?
- Current: {1, 2, 4, 8}
- Alternative: {1, 4, 16}, {2, 4, 8, 16}
- Requires: Analysis of tradeoff

### 11.2 필수 검증 실험

**실험 0** (🔴 최우선): **Policy overhead 측정**
- Policy forward pass FLOPs 정확히 측정
- TTT iteration FLOPs 측정
- Net gain 계산: Savings - Overhead
- **이것이 negative면 프로젝트 중단 고려**

**실험 1**: **Oracle analysis** (Upper bound)
- 각 토큰에 대해 K ∈ {1,2,4,8} 모두 시도
- Best K 분포 확인
- Oracle이 uniform distribution이면 null hypothesis 확인됨
- **Learned policy보다 먼저 수행**

**실험 2**: Convergence analysis
- Measure K-step iterative vs analytical gap
- Find minimum K for <1% gap
- Validate that K=4 or K=8 is sufficient

**실험 3**: Gamma ablation
- Compare γ ∈ {0.0, 0.5, 0.9, 0.99}
- Measure sequential dependency strength
- Justify γ=0.99 choice

**실험 4**: Difficulty correlation
- Compute oracle K for each token
- Measure correlation with entropy, loss, gradient
- Validate difficulty metrics

**실험 5**: Scaling
- WikiText-2 → WikiText-103
- 60M → 250M+ parameters
- Verify findings hold at scale

**실험 순서** (중요):
1. Oracle analysis (feasibility 확인)
2. Policy overhead 측정 (net gain 가능성 확인)
3. Heuristic baselines (간단한 방법으로 baseline 확립)
4. Learned policy (마지막에 시도)

### 11.3 문서화 원칙

**Honest claims only**:
- State what was validated
- Acknowledge what wasn't
- Clear about limitations

**Reproducibility**:
- All hyperparameters documented
- Seeds fixed and reported
- Environment specified

**Comparisons**:
- Clear about what is being compared
- Apples-to-apples (same algorithm variants)
- Fair baselines

---

## 12. 결론

### 핵심 아이디어 요약

PonderTTT는 **"모든 토큰이 동일한 계산을 필요로 하지 않는다"**는 관찰에서 출발하여, **강화학습을 통해 토큰별 최적 gradient descent 반복 횟수를 학습**하는 방법론입니다.

**Three pillars**:
1. **Adaptive allocation**: Per-token K ∈ {1,2,4,8}
2. **REINFORCE learning**: Policy gradient with baseline
3. **Temporal credit**: Monte Carlo returns (γ=0.99)

### 기대 효과 (현실적 평가)

**⚠️ 중요한 면책**: 아래는 이상적 시나리오이며, null result 가능성 높음

**Efficiency (불확실)**:
- 이상적: 5-10% net FLOPs reduction
- 현실적: 0% (policy overhead = savings)
- 최악: Negative (overhead > savings)

**Quality (불확실)**:
- 이상적: Marginal perplexity improvement (< 1%)
- 현실적: No significant difference from Uniform K=4

**Interpretability (확실)**:
- Learned difficulty assessment (성공 여부와 무관)
- 이것만으로도 연구 가치 있음

### 후속 작업 필요

1. ✅ Convergence analysis (iterative vs analytical)
2. ✅ Gamma ablation (validate temporal credit)
3. ✅ Large-scale validation (WikiText-103, larger models)
4. ✅ Honest comparison framework
5. ✅ Statistical rigor (effect sizes, multiple seeds)

---

**이 문서는 PonderTTT의 핵심 아이디어와 방법론을 기록합니다.**
**코드 재작성 시 이 아이디어를 기반으로 하되, 발견된 문제들을 해결한 새로운 구현을 목표로 합니다.**

**⚠️ 중요**: 이 연구는 높은 실패 가능성을 가지고 있습니다. Null/negative result도 학술적으로 가치 있는 기여임을 인정하고 시작해야 합니다.

**마지막 업데이트**: 2025-11-09 (주요 우려사항 반영)

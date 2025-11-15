# PonderTTT: Learning When to Ponder During Test-Time Training
## Adaptive Budget-Constrained Compute Allocation for Code Generation
### Research Plan (v2.0)

---

## Executive Summary

### Core Innovation: 파라미터 적응의 학습된 제어

**PonderTTT**는 테스트 시간 학습(Test-Time Training, TTT)을 위한 최초의 학습된 적응형 정책 프레임워크입니다. 본 연구의 핵심 혁신은 **언제 파라미터를 업데이트할지**를 학습하는 것으로, 이는 추론 경로 선택이나 계산 단계 수 결정과는 근본적으로 다른 문제입니다.

**핵심 차별점 요약:**

| 방법론 | 결정 대상 | 파라미터 변경 | 학습 메커니즘 | 도전과제 |
|--------|----------|-------------|------------|---------|
| **PonderNet** | 계산 단계 수 | ❌ 없음 | 그래디언트 | 정지 시점 결정 |
| **MoD/CALM** | 추론 경로 | ❌ 없음 | 그래디언트 | 효율적 라우팅 |
| **고정 TTT** | N/A | ✅ 있음 (고정) | N/A | 스케줄 설계 |
| **PonderTTT** | 적응 전략 | ✅ 있음 (적응) | RL | 비정상성 + 예산 |

**왜 이것이 새로운 문제인가:**
1. **비정상성(Non-stationarity)**: TTT는 매 업데이트마다 파라미터를 변경 → 정책이 움직이는 타겟에 적응해야 함
2. **비가역성(Irreversibility)**: 잘못된 업데이트는 되돌릴 수 없음 → 탐색 비용이 매우 높음
3. **이중 최적화**: 정책 학습과 동시에 fast-weight 학습 → 불안정성 위험
4. **장기 의존성**: 현재 업데이트가 미래 청크의 난이도에 영향 → 신용 할당 문제

### 주요 특징 (v2.0)

본 계획서의 주요 특징:

1. **평가 전략**: HumanEval/MBPP 외에 ClassEval, repository-level 벤치마크 포함
2. **통계적 엄격성**: Bootstrap CI, IQM 메트릭, 충분한 seed 수
3. **비용 추정**: UPDATE_4 = 12× (4 forward + 8 backward) 정확한 계산
4. **스케일링 접근**: 전이 특성 규명 포함
5. **베이스라인**: 간단한 휴리스틱 및 비-RL 대안 포함
6. **출판 가능성**: 20-35%

### 자원 할당

- **컴퓨팅**: TPU v4-64 (2개월, 확장 가능)
- **예산**: ~$500 (GPU 보조 실험 및 예비 자원)
- **타임라인**: 10개월 → ICLR 2027 (2026년 9월 제출)
- **주요 스케일**: 1B 파라미터 (핵심 결과용)
- **확장 스케일**: 3B-7B (기간 허용 시)

---

## 1. Related Work 및 차별화

### 1.1 적응형 계산 방법론의 분류

적응형 계산(Adaptive Computation)은 크게 세 가지 범주로 나눌 수 있습니다:

#### Category A: 추론 시간 라우팅 (Inference-Time Routing)
**파라미터 변경 없음** - 고정된 모델에서 계산 경로만 선택

| 방법 | 메커니즘 | 결정 대상 | 학습 방법 |
|------|---------|----------|----------|
| **MoE** (Shazeer et al., 2017) | Expert 선택 | 어느 전문가 활성화 | 그래디언트 기반 |
| **CALM** (Schuster et al., 2022) | Early exit | 언제 레이어 중단 | 신뢰도 임계값 |
| **MoD** (Raposo et al., 2024) | Token routing | 어느 토큰 처리 | 그래디언트 (STE) |
| **PonderNet** (Banino et al., 2021) | Step count | 몇 번 반복 | 그래디언트 (β-VAE) |

**공통점**: 모두 **고정된 파라미터**로 추론 효율성 개선
**한계**: 새로운 데이터 분포에 적응 불가능

#### Category B: 테스트 시간 학습 (Test-Time Training)
**파라미터 적응 있음** - 테스트 데이터로 모델 파라미터 업데이트

| 방법 | 업데이트 스케줄 | 적응 메커니즘 | 적용 도메인 |
|------|--------------|-------------|-----------|
| **TTT Layers** (Sun et al., 2020) | 모든 청크 업데이트 | Self-supervised | Vision |
| **MEMO** (Zhang et al., 2022) | 샘플당 고정 | Entropy 최소화 | OOD 일반화 |
| **LaCT** (Zhang et al., 2025) | 모든 청크 업데이트 | LoRA 기반 | Language |
| **TENT** (Wang et al., 2021) | Batch normalization | BN 통계만 | Domain shift |

**공통점**: 모두 **고정된 스케줄** (모든 샘플/청크에 동일 적용)
**한계**: 계산 예산 제약 하에서 최적화 불가능

#### Category C: 적응형 TTT (본 연구)
**학습된 적응 전략** - 언제/얼마나 파라미터를 업데이트할지 학습

| 구성요소 | PonderTTT 접근 | 기존 방법과 차이 |
|---------|--------------|---------------|
| **정책 학습** | RL (PPO + PID) | Category A는 그래디언트, Category B는 학습 없음 |
| **파라미터 업데이트** | 적응형 fast-weight | Category A는 없음, Category B는 고정 |
| **예산 제약** | 엄격한 하드 제약 | 기존 연구는 soft 또는 없음 |
| **입도(Granularity)** | Chunk-level | PonderNet은 token-level |

### 1.2 PonderNet과의 심층 비교

**유사성 (왜 혼동될 수 있는가):**
- 둘 다 "언제 멈출지" 결정하는 문제처럼 보임
- 둘 다 학습된 정책 사용
- 둘 다 예산 제약 고려

**근본적 차이 (왜 완전히 다른 문제인가):**

| 측면 | PonderNet | PonderTTT |
|------|-----------|-----------|
| **불변량** | 파라미터 θ는 고정 | 파라미터 θ는 계속 변함 |
| **상태 공간** | 정상(stationary) | 비정상(non-stationary) |
| **결정 효과** | 계산 비용만 증가 | 모델 자체가 변경됨 |
| **되돌림** | 언제든 가능 | 불가능 (irreversible) |
| **학습 안정성** | 단일 최적화 | 이중 최적화 (정책 + 파라미터) |
| **오류 비용** | 계산 낭비 | 성능 저하 + 계산 낭비 |

**구체적 예시:**

```python
# PonderNet: 고정된 파라미터로 반복 계산
θ = pretrained_model.parameters()  # 고정됨
for step in range(max_steps):
    hidden = f(hidden, θ)  # θ는 불변
    if halting_policy(hidden) > threshold:
        break  # 단순히 멈춤
# θ_final = θ_initial (파라미터 변화 없음)

# PonderTTT: 파라미터를 동적으로 업데이트
θ_slow = pretrained_model.parameters()  # 고정
θ_fast = initialize_fast_weights()       # 적응형
for chunk in sequence:
    features = extract(chunk, θ_fast)
    action = policy(features)  # SKIP or UPDATE_1/2/4
    if action != SKIP:
        θ_fast = gradient_descent(chunk, θ_fast)  # 파라미터 변경!
    output = forward(chunk, θ_slow + θ_fast)
# θ_fast는 시퀀스 전체에 걸쳐 진화
```

**왜 PonderNet 기법이 직접 적용되지 않는가:**

1. **Halting 그래디언트**: PonderNet은 정지 확률 `λ_n`에 대해 직접 그래디언트 계산
   - 전제: 파라미터 θ 고정 → ∂L/∂λ 계산 가능
   - TTT에서는: θ가 λ에 의존 → ∂L/∂λ 계산 불가능 (비미분성)

2. **기하 분포 사전**: PonderNet은 λ ~ Geometric(p) 가정
   - 전제: 각 단계가 독립적 → 단순한 사전 분포
   - TTT에서는: 각 업데이트가 다음 청크에 영향 → 장기 의존성

3. **재구성 손실**: PonderNet은 각 단계의 예측을 평균화
   - 전제: 모든 단계에서 동일한 출력 형식
   - TTT에서는: 업데이트 후 모델이 질적으로 변함 → 평균화 불가능

### 1.3 왜 기존 방법들이 충분하지 않은가

**Q1: 추론 시간 라우팅으로 충분하지 않은가?**

A: 코드 생성에서는 **분포 이동(distribution shift)**이 핵심 도전과제
- 저장소마다 다른 API, 네이밍, 패턴
- 고정된 파라미터는 새로운 패턴에 적응 불가
- 예: React 프로젝트 → Django 프로젝트로 전환 시 라우팅만으로는 한계

**Q2: 고정 스케줄 TTT로 충분하지 않은가?**

A: 코드의 난이도는 **극도로 불균등**
- 보일러플레이트: `import numpy as np` → TTT 불필요 (낭비)
- 복잡한 알고리즘: 다단계 로직 → 더 많은 TTT 필요
- 고정 스케줄: 쉬운 청크에 과투자, 어려운 청크에 과소투자

**Q3: 간단한 휴리스틱으로 충분하지 않은가?**

A: 이것이 본 연구의 핵심 실험 질문 (RQ1)
- Perplexity-based, entropy-based 등 휴리스틱 베이스라인 포함
- 가설: RL이 더 우수할 것 (예산-품질 트레이드오프 학습)
- 만약 휴리스틱이 더 좋다면: 이 또한 중요한 부정적 결과

### 1.4 본 연구의 독특한 기여

**방법론적 기여:**
1. **최초의 학습된 TTT 정책**: 기존 TTT는 모두 고정 스케줄 사용
2. **비정상 환경에서의 RL**: 파라미터 업데이트로 인한 비정상성 해결
3. **예산 제약 RL + TTT**: PID-Lagrangian 기법을 TTT에 적용

**응용적 기여:**
1. **코드 생성에 TTT 최초 적용**: 완전히 새로운 도메인
2. **Repository-level 적응**: 프로젝트별 패턴 학습
3. **실용적 효율성**: 30-40% 계산 절감 목표

**실험적 기여:**
1. **휴리스틱 vs RL 체계적 비교**: 언제 RL이 필요한지 규명
2. **실패 사례 분석**: 어떤 코드 패턴에서 TTT가 실패하는지
3. **스케일링 법칙**: 모델 크기에 따른 정책 전이 특성

---

## 2. 연구 배경 및 동기

### 2.1 Test-Time Training의 필요성

Test-time training은 모델이 테스트 데이터의 지역적 패턴에 적응할 수 있게 하여, 특정 도메인이나 스타일에 대한 성능을 향상시킵니다. 코드 생성에서는:
- 저장소별 코딩 컨벤션
- 프로젝트 특유의 API 사용 패턴
- 일관된 네이밍 및 구조적 패턴

이러한 지역적 패턴들이 TTT의 이상적인 대상이 됩니다.

### 2.2 기존 접근법의 한계

**고정 스케줄 TTT** (LaCT, TTT Layers):
```python
# 모든 청크에 동일한 업데이트 적용
for chunk in sequence:
    θ_fast = update_fast_weights(chunk)  # 항상 업데이트
```

이 접근법의 문제점:
- 쉬운 청크(보일러플레이트 코드)에 계산 낭비
- 어려운 청크(복잡한 알고리즘)에 자원 부족
- 예산 제약 하에서 최적화 불가능

### 2.3 PonderTTT 솔루션

```python
# 학습된 정책으로 적응형 할당
for chunk in sequence:
    features = extract_features(chunk, model_state)
    action = policy(features, budget_remaining)
    θ_fast = apply_action(action, chunk)  # SKIP or UPDATE_1/2/4
```

**액션 스페이스**:
- `SKIP`: 업데이트 없음 (1× forward)
- `UPDATE_1`: 1회 그래디언트 스텝 (3× = 1 fwd + 2 bwd)
- `UPDATE_2`: 2회 그래디언트 스텝 (5× = 2 fwd + 4 bwd)
- `UPDATE_4`: 4회 그래디언트 스텝 (12× = 4 fwd + 8 bwd)

**비용 계산 근거**: 역전파는 전방향 전달의 2배 비용 (가중치 그래디언트 계산 + 오차 역전파)

---

## 3. 연구 질문

### RQ1: 효율성 - 학습된 정책 vs 베이스라인
> 학습된 정책이 고정 스케줄, 휴리스틱, 그리고 PonderNet-스타일 베이스라인 대비 품질-FLOPs Pareto 프론티어에서 우수한가?

**베이스라인 계층:**

**Tier 1: 기본 (No Learning)**
1. **No-TTT**: 사전학습 모델만 사용 (하한선)
2. **Fixed-All**: 모든 청크에 UPDATE_2 적용
3. **Fixed-Schedule**: 매 N 청크마다 UPDATE_2

**Tier 2: 휴리스틱 (Simple Rules)**
4. **Perplexity-Based**: 높은 perplexity 청크만 업데이트
5. **Entropy-Based**: 예측 엔트로피 > 임계값
6. **Gradient-Norm**: 큰 그래디언트 → 더 많은 업데이트

**Tier 3: PonderNet-스타일 (Gradient-Based Learning)**
7. **Halting-Policy**: PonderNet 기법을 TTT에 적용 시도
   - 정지 확률 학습 (λ_n)
   - 재가중 손실: L = Σ λ_n * L_n
   - **예상 실패 원인**: 비정상성으로 인한 불안정
   - **실험 목적**: 왜 그래디언트 기반 방법이 TTT에 부적합한지 실증

**Tier 4: RL-Based (Our Method)**
8. **PonderTTT**: RL 정책 (PPO + PID)
9. **Oracle**: 사후 분석으로 최적 액션 선택 (상한선)

**가설**:
- PonderTTT는 Tier 1-2보다 명확히 우수 (30-40% 계산 절감 또는 10-15% 품질 향상)
- Halting-Policy (Tier 3)는 학습 불안정으로 실패할 것
- RL이 필요한 이유: 비정상성 + 장기 의존성 처리

### RQ2: 해석가능성
> 학습된 정책이 오라클 분석으로 측정한 진정으로 어려운 청크에 계산을 집중하는가?

**가설**: 정책 결정과 오라클 식별 어려운 청크 간 상관관계 (Spearman ρ > 0.6) 

### RQ3: 일반화
> 정책이 (a) 계산 예산, (b) 청크 크기, (c) 저장소, (d) 모델 스케일 간 어떻게 일반화/적응하는가?

**가설**: 
- 동일 스케일 내 예산/청크 크기 전이: >80% 성능 유지 가능
- 크로스 스케일 전이: 특성 규명이 주요 목표, 절대 전이는 부차적
- RAST (2025) 연구에서 32B 모델에 14B 정책 적용 시 95% 성능 회복 사례 존재
- 본 연구는 이러한 전이 특성을 TTT 맥락에서 체계적으로 분석

### RQ4: 도메인 유효성
> TTT가 코드 생성에 효과적인가? 어떤 코드 패턴에서 가장 큰 이득이 있는가?

**가설**: 구조적 복잡성과 API 밀도가 높은 코드에서 최대 15-20% 향상

### RQ5: 실패 사례 분석 (신규)
> 어떤 상황에서 PonderTTT가 실패하거나 휴리스틱만큼만 성능을 내는가?

**분석 축:**

**A. 코드 특성별 성능 분해**
```python
# 각 카테고리에서 PonderTTT vs Perplexity-Based 비교
categories = {
    "boilerplate": ["import", "class definition", "simple assignments"],
    "algorithmic": ["sorting", "search", "dynamic programming"],
    "API-heavy": ["library calls", "framework usage"],
    "edge-cases": ["error handling", "corner cases"],
    "creative": ["novel solutions", "uncommon patterns"]
}
```

**예상 결과:**
- Boilerplate: 휴리스틱과 동등 (간단한 규칙으로 충분)
- Algorithmic: PonderTTT 우수 (복잡한 트레이드오프 학습)
- API-heavy: PonderTTT 크게 우수 (TTT가 API 패턴 학습)
- Edge-cases: 양쪽 모두 어려움 (데이터 희소성)
- Creative: 불확실 (경험적으로 규명)

**B. 실패 모드 분류**

| 실패 유형 | 증상 | 원인 | 비율 예상 |
|---------|------|------|---------|
| **Over-adaptation** | 너무 많은 UPDATE | RL 과학습 | 10-15% |
| **Under-adaptation** | 너무 많은 SKIP | 보수적 정책 | 5-10% |
| **Catastrophic update** | 업데이트 후 성능 저하 | 나쁜 그래디언트 | 5-10% |
| **Budget misallocation** | 쉬운 청크에 낭비 | 특징 부족 | 10-20% |
| **No-gain regime** | TTT 자체가 무효 | 작은 분포 이동 | 20-30% |

**C. 정량적 메트릭**

```python
# 각 테스트 샘플에 대해 계산
metrics = {
    "regret": oracle_score - ponderttt_score,  # 얼마나 최적에서 멀었나
    "over_budget": actual_cost - budget_limit,  # 예산 위반
    "policy_variance": std(action_logits),      # 정책 확신도
    "feature_drift": KL(p_train || p_test)      # 분포 이동 정도
}

# 실패 케이스: regret > threshold
failure_cases = samples[regret > 0.2 * oracle_score]
```

**D. 정성적 분석**

**Case Study 1: 성공 케이스**
- 코드: Django ORM 쿼리 생성
- 관찰: PonderTTT가 `.filter()` 패턴에 UPDATE_4 집중
- 분석: TTT가 프로젝트 특화 쿼리 패턴 학습
- 결론: Repository-level 적응의 증거

**Case Study 2: 실패 케이스**
- 코드: 간단한 헬퍼 함수
- 관찰: PonderTTT가 불필요하게 UPDATE_2 선택
- 분석: 특징이 난이도 과대평가
- 결론: 특징 엔지니어링 개선 필요

**Case Study 3: 휴리스틱 동등**
- 코드: 표준 라이브러리 사용
- 관찰: Perplexity-Based와 동일한 결정
- 분석: 명확한 난이도 시그널
- 결론: RL이 불필요한 영역 (간단한 규칙으로 충분)

**E. 학습 곡선 분석**

```python
# 정책 학습 중 추적
training_phases = {
    "early": (0, 20),      # 탐색 단계
    "mid": (20, 60),       # 수렴 단계
    "late": (60, 100)      # 안정화 단계
}

for phase in training_phases:
    analyze_failure_modes(phase)
    # 가설: early는 random 실패, late는 systematic 실패
```

**F. 한계 인정 및 미래 연구**

**알려진 한계:**
1. **Chunk-level granularity**: Token-level보다 덜 세밀
   - 완화: 계산 효율성 vs 정밀도 트레이드오프
   - 미래: 적응형 청크 크기

2. **Self-supervised 태스크 의존성**: 태스크 품질이 중요
   - 완화: 다양한 태스크 시도 (MLM, NSP, code-specific)
   - 미래: 메타 학습으로 태스크 선택

3. **분포 이동 필요**: 유사한 코드에서는 이득 미미
   - 완화: Repository-level 평가로 실제 시나리오 측정
   - 미래: 분포 이동 탐지 메커니즘

4. **RL 샘플 복잡도**: 많은 학습 데이터 필요
   - 완화: 오프라인 RL, 휴리스틱 사전학습
   - 미래: few-shot 적응

**출판 전략:**
- 실패 사례를 숨기지 않고 투명하게 보고
- "When does adaptive TTT help?" 프레이밍
- 부정적 결과도 학술적 가치 강조
- 커뮤니티가 피할 수 있는 함정 공유

---

## 4. 방법론

### 4.1 강화학습 프레임워크

**알고리즘**: PID-Lagrangian PPO (Stooke et al., 2020)

**주의사항**: 이 방법은 2020년 제안된 혁신적 기법으로, "확립된(well-established)" 방법이라기보다는 최신 기법입니다. 본 연구에서는:
- 철저한 하이퍼파라미터 튜닝 수행
- 제약 위반 모니터링 시스템 구축
- 대안 방법(Penalty-based PPO) 준비

**상태 공간**: 32차원 특징 벡터
- 활성화 통계 (평균, 표준편차, 스파시티)
- 어텐션 패턴 (엔트로피, 범위)
- 코드 특화 메트릭 (토큰 엔트로피, OOV 비율, 순환 복잡도)
- 역사적 맥락 (최근 난이도 EMA, 남은 예산)
- 모델 신뢰도 (예측 엔트로피, perplexity)

**보상 함수**:
```python
reward = quality_improvement - λ * (cost_used / budget)
```
- λ는 PID 컨트롤러로 동적 조정
- 즉각적 손실 감소도 부분 보상에 포함 (학습 효율성)

### 4.2 Fast-Weight 업데이트 메커니즘

**아키텍처**: LaCT-스타일 low-rank adaptation
- LoRA rank: 64/128/256 (ablation 예정)
- 업데이트 대상: 어텐션 레이어의 query/value 프로젝션
- 이론적 근거: 70% GPU 활용률 달성 (Zhang et al., 2025)

### 4.3 특징 추출

**설계 원칙**:
- <1% 오버헤드 (캐시된 활성화 활용)
- 해석 가능한 특징 (ablation 연구 가능)
- 코드 특화 시그널 포함

**Ablation 계획**: 각 특징 그룹의 중요도 측정 (5개 카테고리)

### 4.4 PonderNet-스타일 베이스라인 구현 (신규)

**목적**: 왜 그래디언트 기반 방법이 TTT에 부적합한지 실증적으로 보여주기

**구현 전략:**

```python
# Halting-Policy: PonderNet을 TTT에 적용
class HaltingTTTPolicy:
    def __init__(self):
        self.halting_network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # λ_n ∈ [0, 1]
        )

    def forward(self, features, chunk_idx):
        # 정지 확률 계산
        lambda_n = self.halting_network(features)

        # 액션 샘플링 (확률적)
        action_probs = self.get_action_distribution(lambda_n, chunk_idx)
        action = sample(action_probs)

        return action, lambda_n

    def loss(self, outputs, labels, lambdas):
        # 재가중 손실 (PonderNet-style)
        weighted_loss = sum(lambda_n * loss_n
                           for lambda_n, loss_n in zip(lambdas, losses))

        # 정규화 항 (계산 비용 페널티)
        cost_penalty = beta * sum(lambdas)

        # KL 발산 (기하 분포 사전)
        kl_term = KL(lambdas || Geometric(p))

        return weighted_loss + cost_penalty + kl_term
```

**예상 문제점:**

1. **비정상성 문제**:
   ```python
   # θ_fast가 변하면서 특징 분포도 변함
   features_t = extract(chunk_t, θ_fast_t)
   θ_fast_{t+1} = update(θ_fast_t)  # 파라미터 변경
   features_{t+1} = extract(chunk_{t+1}, θ_fast_{t+1})  # 다른 분포!

   # 결과: halting_network의 입력 분포가 학습 중 계속 변함
   # → 그래디언트가 불안정
   ```

2. **비미분성 문제**:
   ```python
   # TTT 업데이트는 그래디언트 차단됨
   θ_fast = gradient_descent(loss, θ_fast).detach()  # 명시적 차단

   # 결과: ∂output/∂λ 계산 시 θ_fast의 변화를 고려 못함
   # → 편향된 그래디언트
   ```

3. **장기 의존성 문제**:
   ```python
   # 현재 업데이트가 미래 청크에 영향
   action_t = policy(chunk_t)
   if action_t == UPDATE:
       θ_fast = update(θ_fast)
       # 이제 chunk_{t+1}, chunk_{t+2}, ... 모두 영향받음

   # 결과: 신용 할당(credit assignment) 매우 어려움
   # → 그래디언트 기반 방법으로 해결 곤란
   ```

**실험 계획:**

| 메트릭 | 예상 결과 | 의미 |
|--------|---------|------|
| **수렴 안정성** | 발산 또는 수렴 실패 | 비정상성의 증거 |
| **최종 성능** | 휴리스틱보다 나쁨 | 편향된 그래디언트 |
| **예산 준수** | 위반 빈번 | KL 정규화 불충분 |
| **학습 곡선** | 진동 또는 불안정 | 이중 최적화 문제 |

**성공 시나리오 (unlikely but possible):**
- 만약 Halting-Policy가 작동한다면: 더 간단한 방법 선호 (Occam's razor)
- 본 연구의 기여: RL vs 그래디언트 비교 → 언제 어느 방법이 적합한지 규명

**실패 시나리오 (expected):**
- Halting-Policy 실패 → 왜 RL이 필요한지 정당화
- 논문에서 강조: TTT의 독특한 도전과제 (비정상성, 비미분성, 장기 의존성)

---

## 5. 실험 설계

### 5.1 벤치마크 전략 (3-계층 접근)

#### Tier 1: 표준 벤치마크 (비교 가능성)
- **HumanEval** (164 문제): 기준선 비교용
- **MBPP** (974 문제): 추가 검증
- **MultiPL-E**: 크로스 언어 일반화

**인정**: 이 벤치마크들은 작고 context-free한 알고리즘 문제로 구성되어 있어, 저장소 수준의 패턴 학습을 완전히 평가하기 어렵습니다.

#### Tier 2: 클래스 수준 평가 (신규)
- **ClassEval** (100 클래스): 45.7줄 평균, 76% 메서드가 클래스 내 의존성 보유
- 더 긴 컨텍스트와 지역적 일관성 요구
- TTT의 적응 능력을 더 잘 측정

#### Tier 3: Repository-Level 평가 (신규)
- **사용자 정의 벤치마크**: The Stack에서 선별한 10개 저장소
  - 파일당 평균 300+ 줄
  - 크로스 파일 의존성 포함
  - 일관된 코딩 스타일 보유
- **평가 방식**: 
  - 저장소의 80%로 TTT 수행
  - 나머지 20%에서 함수 생성 평가
  - 저장소 특화 패턴 학습 능력 측정

**근거**: ClassEval 논문 (Du et al., 2023)과 REPOCOD (Nam et al., 2024)가 repository-level 평가의 중요성 입증

### 5.2 베이스라인

**기본 베이스라인**:
1. No-TTT: 사전학습 모델만
2. Fixed-All: 모든 청크에 동일 업데이트
3. Fixed-Schedule: 매 N 청크마다 업데이트

**적응형 베이스라인** (신규):
4. **Perplexity-Based**: 높은 perplexity 청크만 업데이트 (간단한 휴리스틱)
5. **Entropy-Based**: 예측 엔트로피 기반 선택
6. **Random-Budget**: 무작위 선택 (예산 제약 하)

**오라클 상한**:
7. Oracle: 사후 분석으로 실제 이득이 큰 청크만 업데이트

**중요성**: 간단한 휴리스틱보다 RL이 우수함을 입증해야 출판 가치 확보

### 5.3 모델 스케일 및 실험 설계

| 스케일 | 목적 | Seeds | 통계 |
|--------|------|-------|------|
| 125M | 초기 검증, 빠른 반복 | 10 | Bootstrap CI, IQM |
| 350M | 방법론 확정, ablation | 7 | Bootstrap CI, IQM |
| 1B | **핵심 결과** | 5 | Bootstrap CI, IQM, 완전한 ablation |
| 3B | 확장 목표 (기간 허용 시) | 3 | Bootstrap CI, IQM |
| 7B | 최대 목표 (예비) | 3 | Bootstrap CI, IQM |

**통계적 엄격성**:
- **Bootstrap Confidence Intervals**: 1000회 resampling
- **Interquartile Mean (IQM)**: 이상치에 강건한 메트릭
- **Performance Profiles**: 전체 분포 시각화
- **통계적 유의성 검정**: paired t-test 또는 Wilcoxon signed-rank

**근거**: "Deep RL at the Edge of the Statistical Precipice" (Agarwal et al., NeurIPS 2021 Outstanding Paper)의 권고사항 반영

### 5.4 평가 메트릭

**주요 메트릭**:
- **pass@k** (k=1,10,100): 코드 정확성
- **FLOPs / wall-clock time**: 계산 비용
- **Pareto frontier**: 품질-비용 트레이드오프
- **AUC (Area Under Curve)**: Pareto 곡선 면적

**보조 메트릭**:
- Policy entropy: 탐색 vs 활용
- Feature importance: Ablation 결과
- Update frequency: 액션 분포
- Correlation with oracle: ρ > 0.6 목표

### 5.5 실패 사례 분석 실험 설계 (신규)

**목적**: RQ5 (실패 사례 분석)를 체계적으로 수행하기 위한 실험 프로토콜

**Phase A: 데이터 수집 (자동)**

```python
# 모든 테스트 샘플에 대해 자동 수집
for sample in test_set:
    results[sample.id] = {
        # 각 방법의 결과
        "no_ttt": evaluate(sample, NoTTT()),
        "fixed": evaluate(sample, FixedTTT()),
        "perplexity": evaluate(sample, PerplexityBased()),
        "halting": evaluate(sample, HaltingPolicy()),  # 예상: 실패
        "ponderttt": evaluate(sample, PonderTTT()),
        "oracle": evaluate(sample, Oracle()),

        # 코드 특성
        "code_features": {
            "category": classify_code_type(sample),  # 5개 카테고리
            "complexity": cyclomatic_complexity(sample),
            "api_density": count_api_calls(sample) / len(sample),
            "has_loops": detect_loops(sample),
            "novelty": measure_novelty(sample, training_set),
        },

        # 정책 행동
        "action_sequence": record_actions(sample, PonderTTT()),
        "budget_used": measure_cost(sample, PonderTTT()),
        "feature_values": extract_features(sample),

        # 오류 분석
        "failure_mode": classify_failure(sample),  # 5가지 유형
        "regret": oracle_score - ponderttt_score,
    }
```

**Phase B: 분류 및 클러스터링 (자동)**

```python
# 1. 코드 카테고리별 성능 분해
for category in ["boilerplate", "algorithmic", "api-heavy", "edge-cases", "creative"]:
    subset = filter_by_category(results, category)
    compare_methods(subset)  # Wilcoxon signed-rank test
    visualize_pareto_frontier(subset)

# 2. 실패 케이스 클러스터링
failure_cases = [s for s in results if s["regret"] > threshold]
clusters = kmeans(
    features=[s["code_features"] for s in failure_cases],
    n_clusters=5
)

for cluster in clusters:
    print(f"Cluster {cluster.id}:")
    print(f"  Size: {len(cluster.samples)}")
    print(f"  Common features: {cluster.centroid}")
    print(f"  Dominant failure mode: {cluster.mode}")

# 3. 특징-성능 상관관계
correlations = {}
for feature in all_features:
    correlations[feature] = spearman(
        feature_values=results[feature],
        performance=results["regret"]
    )
```

**Phase C: 정성적 분석 (수동)**

```python
# 각 클러스터에서 대표 샘플 선택
representative_samples = {
    "success": select_top_k(results, key="regret", k=10, ascending=True),
    "failure": select_top_k(results, key="regret", k=10, ascending=False),
    "heuristic_equivalent": select_near_zero_regret(results, k=10),
}

# 수동 검토 및 Case Study 작성
for sample in representative_samples:
    manual_analysis = {
        "code": sample.code,
        "observation": describe_policy_behavior(sample),
        "analysis": explain_why(sample),
        "conclusion": derive_insight(sample),
    }
    write_case_study(manual_analysis)
```

**Phase D: 시각화**

1. **코드 카테고리별 성능 분해**:
   ```
   [Bar chart: 5 categories × 4 methods (Fixed, Perplexity, Halting, PonderTTT)]
   Y축: Pass@1 improvement over No-TTT
   색상: 통계적 유의성 표시
   ```

2. **실패 모드 분포**:
   ```
   [Pie chart: 5 failure modes]
   - Over-adaptation: 15%
   - Under-adaptation: 10%
   - Catastrophic update: 8%
   - Budget misallocation: 17%
   - No-gain regime: 50%
   ```

3. **특징-성능 상관관계**:
   ```
   [Heatmap: features × performance metrics]
   값: Spearman ρ
   유의성: 별표 표시 (*, **, ***)
   ```

4. **학습 곡선별 실패 모드**:
   ```
   [Line plot: training iterations × failure mode prevalence]
   3 curves: early (0-20), mid (20-60), late (60-100)
   ```

**Phase E: 논문 작성 가이드**

**Main Paper 섹션:**
- 5.5 "Failure Mode Analysis" (1 페이지)
  - Table: 코드 카테고리별 성능 분해
  - Figure: 실패 모드 분포
  - 2-3개 Case Study (성공, 실패, 동등)

**Appendix 섹션:**
- A.3 "Complete Failure Analysis" (3-4 페이지)
  - 모든 클러스터 상세 분석
  - 추가 Case Study (5-10개)
  - 특징-성능 상관관계 전체 표
  - 학습 곡선 분석
  - 한계 및 미래 연구 방향

**예상 Reviewer 질문 대응:**

Q: "간단한 휴리스틱이 RL만큼 좋다면, 왜 복잡한 방법을 쓰나?"
A: "실패 분석에서 코드의 50%는 휴리스틱과 동등, 30%는 RL이 우수, 20%는 둘 다 실패. 복잡한 알고리즘과 API-heavy 코드에서 RL의 가치 입증. (Section 5.5, Figure X)"

Q: "실패 사례가 많은데, 이 방법이 실용적인가?"
A: "No-gain regime (50%)은 TTT 자체의 한계 (분포 이동 부족). PonderTTT의 실패는 8% 미만 (catastrophic update). 나머지는 성공 또는 동등. (Table X, Appendix A.3)"

Q: "PonderNet이 왜 실패하는지 증명했나?"
A: "Halting-Policy 베이스라인이 수렴 실패 및 예산 위반 (Section 5.2, Figure Y). 비정상성으로 인한 그래디언트 불안정 확인. RL이 필요한 이유 실증."

---

## 6. 타임라인 (10개월)

### Phase 1: 기반 구축 (3개월)

**Week 1-2: 인프라**
- TPU v4-64 설정 및 검증
- 데이터 파이프라인 구축 (The Stack)
- 베이스라인 구현 (125M)

**Week 3-4: 초기 검증**
- 125M 모델 TTT 검증
- 특징 추출 오버헤드 확인 (<1%)
- GO/NO-GO 체크포인트 #1

**Week 5-8: 방법론 개발**
- PID-Lagrangian PPO 구현
- 정책 네트워크 학습 (125M)
- 초기 결과 분석

**Week 9-12: Ablation 연구**
- 특징 그룹 ablation
- LoRA rank 튜닝
- 350M 스케일업
- GO/NO-GO 체크포인트 #2

### Phase 2: 핵심 실험 (5개월)

**Month 4-5: 1B 스케일 실험**
- 5 seeds × 모든 베이스라인
- 전체 ablation 연구
- ClassEval 평가
- Repository-level 평가 시작

**Month 6: 분석 및 정제**
- 통계 분석 (Bootstrap CI, IQM)
- 정책 해석가능성 분석
- 추가 실험 (gap filling)
- GO/NO-GO 체크포인트 #3

**Month 7-8: 스케일업 (선택적)**
- 3B 모델 (기간 허용 시)
- 7B 모델 (예비)
- 크로스 스케일 전이 분석

### Phase 3: 논문 작성 (2개월)

**Month 9: 초고 작성**
- 방법론, 실험, 결과 섹션
- 그림 및 표 생성
- Related work 섹션

**Month 10: 최종화**
- 내부 리뷰
- Appendix 작성
- 코드 및 체크포인트 준비
- ICLR 2027 제출 (2026년 9월)

---

## 7. 리스크 관리

### 7.1 기술적 리스크

**Risk 1: RL 학습 불안정**
- *발생 확률*: 40%
- *영향도*: 높음
- *완화 전략*:
  - 철저한 하이퍼파라미터 서치
  - Hard budget limits 구현
  - 대안: Penalty-based PPO (P3O)
  - 최종 대안: Supervised learning으로 휴리스틱 정책 학습

**Risk 2: TTT 이득 미미**
- *발생 확률*: 30%
- *영향도*: 높음
- *완화 전략*:
  - 초기 125M 검증 (Week 4 체크포인트)
  - 다양한 self-supervised 태스크 시도
  - Pivot: Repository-level 평가에 집중
  - 최종 대안: 추론 시간 계산 할당으로 전환

**Risk 3: 간단한 휴리스틱이 더 우수**
- *발생 확률*: 25%
- *영향도*: 중간
- *완화 전략*:
  - 포괄적 베이스라인 평가
  - RL의 장기 적응 능력 강조
  - 부정적 결과도 학술적 기여로 인정
  - 논문 프레이밍: "When does RL help?"

### 7.2 자원 리스크

**Risk 4: TPU 시간 부족**
- *발생 확률*: 20%
- *영향도*: 중간
- *완화 전략*:
  - 우선순위: 1B 스케일에 집중
  - 백업: Vast.ai GPU 임대 (~$500)
  - 타임라인 조정: 3B/7B 생략

### 7.3 출판 리스크

**Risk 5: ICLR 리젝**
- *발생 확률*: 65-80% (현실적 추정)
- *영향도*: 중간
- *완화 전략*:
  - 제출 시 코드 및 체크포인트 공개
  - 광범위한 appendix (모든 실험 데이터)
  - 백업 학회: NeurIPS 2027, ICML 2027
  - 워크샵 제출: CVPR/ICML Test-Time Adaptation

---

## 8. 예상 기여

### 8.1 방법론적 기여

1. **최초의 TTT 아키텍처 결정을 위한 학습 정책 프레임워크**
   - 추론 시간 라우팅과 명확히 구분
   - PonderNet과 차별화 (TTT 결정 vs 계산 단계)
   
2. **예산 제약 RL을 TTT에 적용한 최초 사례**
   - PID-Lagrangian 최적화로 보장된 예산 준수
   - 품질-비용 Pareto 프론티어 생성

3. **청크 수준 정책의 효율성 입증**
   - Token-level (PonderNet)보다 효율적
   - Fixed-schedule (LaCT)보다 적응적

### 8.2 응용적 기여

1. **코드 생성에 대한 TTT의 최초 적용**
   - 완전히 새로운 도메인 개척
   - 자가 지도 학습 태스크 개발
   
2. **3-계층 평가 프레임워크**
   - 표준 벤치마크 (비교 가능성)
   - 클래스 수준 (중간 복잡도)
   - Repository-level (실제 시나리오)

3. **실용적 영향**
   - 프로덕션 코드 생성에서 30-40% 계산 절감
   - IDE 통합 가능성

### 8.3 경험적 기여

1. **체계적 스케일링 연구** (125M → 7B)
   - 정책 전이 특성 규명
   - 스케일별 최적 전략 제시
   
2. **철저한 Ablation 연구**
   - 어떤 특징이 중요한가?
   - 어떤 코드 패턴에서 TTT가 효과적인가?
   - 간단한 휴리스틱 vs RL 비교

3. **오픈소스 구현**
   - JAX/Flax 최적화 코드
   - 재현 가능한 실험
   - 커뮤니티 기여

---

## 9. 성공 기준

### 9.1 최소 성공 (출판 가능)

**1B 스케일 결과만으로도:**
- HumanEval/MBPP에서 베이스라인 대비 명확한 개선
- 통계적으로 유의미한 차이 (p < 0.05)
- ClassEval 또는 Repository-level에서 강력한 결과
- 완전한 ablation 및 분석

**출판 가능성**: 20-35% (ICLR 2027)

### 9.2 강력한 성공

**1B + 3B 결과:**
- 모든 3개 벤치마크 계층에서 일관된 개선
- 30-40% 계산 절감 또는 10-15% 품질 향상
- 간단한 휴리스틱보다 명확히 우수
- 스케일 전이 특성 규명

**출판 가능성**: 35-50% (ICLR 2027)

### 9.3 탁월한 성공

**1B + 3B + 7B 결과:**
- 7B에서 SOTA TTT 방법 초과
- Repository-level에서 20% 이상 향상
- 크로스 스케일 정책 전이 성공
- 실용적 배포 시나리오 제시

**출판 가능성**: 40-55% (spotlight 후보)

---

## 10. 결론

### 10.1 왜 이 연구가 성공할 것인가

**기술적 건전성**:
- 입증된 RL 알고리즘 (PID-Lagrangian PPO)
- 효율적 TTT 아키텍처 (LaCT 기반)
- 현실적 계산 비용 추정
- 철저한 베이스라인 비교

**새로움 보존**:
- 코드 생성에 대한 TTT 최초 적용 확인됨
- 추론 시간 방법들과 명확한 차별화
- 학술적 포지셔닝 검증됨

**자원 적절성**:
- TPU v4-64는 1B-7B 실험에 충분
- 10개월 타임라인은 여유 있음
- 다중 go/no-go 게이트로 조기 pivot 가능

**리스크 관리**:
- 각 주요 리스크에 대한 완화 전략
- 백업 컴퓨팅 및 출판 옵션
- 현실적 기대치 설정

### 10.2 현실적 기대

**출판**: 20-35% (ICLR 2027)
- 리젝 가능성이 더 높지만, 백업 계획 준비됨
- 워크샵 제출로 커뮤니티 피드백 조기 확보
- NeurIPS/ICML 2027 대안 확보

**기술적 성공**: 70-80% 확률
- 125M 검증 단계에서 조기 신호 확인
- Pivot 전략으로 실패 리스크 완화

**학술적 기여**: 보장됨
- 실패해도 부정적 결과가 학습이 됨
- 코드 생성 + TTT 조합은 새로운 영역
- 오픈소스 기여로 커뮤니티 가치 창출

### 10.3 즉시 착수 단계

**Day 1-3: TPU 설정**
```bash
gcloud compute tpus tpu-vm create ponderttt-v4-64 \
  --zone=us-central2-b \
  --accelerator-type=v4-64 \
  --version=tpu-ubuntu2204-base
```

**Day 4-7: 데이터 준비**
```python
# The Stack Python subset
dataset = load_dataset("bigcode/the-stack-dedup", 
                       data_dir="data/python",
                       split="train", streaming=True)
chunks = preprocess_and_chunk(dataset, chunk_size=4096)
```

**Week 2: 베이스라인 검증**
- 125M 모델 훈련
- No-TTT 베이스라인 확립
- 목표 perplexity 달성 확인

**Week 3: GO/NO-GO #1**
- TTT 작동 여부 확인
- 특징 추출 오버헤드 검증
- 다음 단계 진행 또는 pivot 결정

---

## 부록 A: 참고문헌

1. **PID-Lagrangian PPO**: Stooke et al., "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods", ICML 2020
2. **LaCT**: Zhang et al., "Test-Time Training Done Right", arXiv:2505.23884, 2025
3. **PonderNet**: Banino et al., "PonderNet: Learning to Ponder", ICML 2021
4. **ClassEval**: Du et al., "ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation", ICSE 2024
5. **Statistical Rigor**: Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021
6. **RAST**: "Reasoning Activation in LLMs via Small-model Transfer", arXiv:2506.15710, 2025
7. **HumanEval**: Chen et al., "Evaluating Large Language Models Trained on Code", arXiv:2107.03374, 2021
8. **MBPP**: Austin et al., "Program Synthesis with Large Language Models", arXiv:2108.07732, 2021


# Inverse Optimal Control for MPC Parameter Learning

## 개요

이 프로젝트는 **KKT 조건 완화(Relaxed KKT Conditions)**를 이용한 역최적제어(Inverse Optimal Control)로 MPC 비용함수 파라미터를 자동으로 학습합니다.

### 핵심 아이디어

```
전문가 시연 데이터
    ↓
역최적제어 (KKT 완화)
    ↓
MPC 비용함수 가중치
    ↓
개선된 제어 성능
```

## 설치 및 요구사항

```bash
# Python 패키지
pip install numpy scipy matplotlib torch mujoco

# MuJoCo 모델 필요
```

## 사용 방법

### Step 1: 시연 데이터 생성

원래 MPC 파라미터로 시뮬레이션을 실행하여 "전문가 시연" 데이터를 수집합니다.

```bash
python main.py
```

**출력:**
- `delta_tau_dataset.npz`: 시연 데이터 (상태, 제어 입력)
- `result_mpc_only.npz`: 성능 메트릭 (비교용)

### Step 2: MPC 파라미터 학습

시연 데이터로부터 KKT 조건 완화를 이용하여 최적 비용함수 가중치를 학습합니다.

```bash
python learn_mpc_weights.py
```

**수행 과정:**
1. 시연 데이터 로드 및 세그먼트 분할
2. 목적함수 정의: `min Σ‖∇L(θ)‖²`
3. SLSQP 최적화로 θ 탐색
4. 학습된 파라미터 저장

**출력:**
- `learned_mpc_weights.npz`: 학습된 가중치
- `learned_weights_config.py`: Python config 파일
- `ioc_results.png`: 시각화 결과

### Step 3: 학습된 파라미터로 MPC 실행

학습된 가중치를 적용하여 시뮬레이션을 다시 실행하고 성능을 비교합니다.

```bash
python apply_learned_mpc.py
```

**출력:**
- `result_mpc_learned.npz`: 학습된 가중치의 성능
- 원래 가중치 vs 학습된 가중치 비교표

## 파일 구조

```
.
├── inverse_optimal_control.py   # 핵심 IOC 알고리즘
├── learn_mpc_weights.py         # 학습 실행 스크립트
├── apply_learned_mpc.py         # 학습된 가중치 적용
├── README_IOC.md                # 이 문서
│
├── dataGet/
│   ├── main.py                  # 원래 MPC 실행
│   ├── mpc_controller.py        # MPC 컨트롤러
│   ├── config.py                # 설정 파일
│   └── ...
│
└── outputs/
    ├── learned_mpc_weights.npz
    ├── ioc_results.png
    └── ...
```

## 주요 클래스 및 함수

### `InverseOptimalControl`

역최적제어의 핵심 클래스입니다.

```python
from inverse_optimal_control import InverseOptimalControl

# IOC 객체 생성
ioc = InverseOptimalControl(
    model=mujoco_model,
    joint_ids=[0, 1, 2],
    horizon=20,
    dt=0.005
)

# 시연 데이터 로드
demos = ioc.load_demonstration_data("delta_tau_dataset.npz")

# 가중치 학습
theta_learned, result = ioc.learn_cost_weights(demos)
```

**주요 메서드:**

- `load_demonstration_data(path)`: 시연 데이터 로드
- `compute_gradient_norm(theta, demo)`: ‖∇L‖² 계산
- `learn_cost_weights(demos, theta_init)`: 최적화 실행
- `compare_parameters(theta_learned, theta_original)`: 파라미터 비교
- `visualize_results(theta_learned, demos)`: 결과 시각화

### `apply_learned_weights_to_mpc()`

학습된 가중치를 MPC 컨트롤러에 적용합니다.

```python
from inverse_optimal_control import apply_learned_weights_to_mpc

apply_learned_weights_to_mpc(mpc_controller, theta_learned)
```

## 수학적 배경

### 문제 정식화

전문가 시연 `U_t = [u(0), ..., u(N)]`가 주어졌을 때, 어떤 비용함수를 최적화했는지 역으로 추정합니다.

**비용함수 파라미터화:**
```
l(x, u, θ) = θ₁·‖q - q_ref‖² + θ₂·‖qdot‖² + θ₃·‖u‖²
```

**목표:** θ = [θ₁, θ₂, θ₃, θ₄] 찾기

### KKT 조건

최적 제어 `U*`는 다음 KKT 조건을 만족해야 합니다:

```
∇_U L(U, θ)|_{U=U_t} = 0    (Stationarity)
λᵀg(U_t) = 0                 (Complementarity)
λ ≥ 0                        (Dual Feasibility)
```

여기서 `L = 비용함수 + λᵀ·제약조건`

### KKT 조건 완화

정확한 조건 만족이 어려우므로 다음 문제를 풉니다:

```
minimize_{θ}  ‖∇_U L(U, θ)|_{U=U_t}‖²

subject to:   θ ≥ 0
              Σθᵢ = const  (정규화)
```

**직관:**
- 기울기를 정확히 0으로 만들 수 없다면
- 기울기를 최대한 0에 가깝게!

## 결과 해석

### 학습 성공 지표

1. **Optimization Success**: `True`여야 함
2. **Final ‖∇L‖²**: 작을수록 좋음 (< 1e-6 권장)
3. **Parameter Changes**: 합리적인 범위 (0.1x ~ 10x)

### 성능 비교

`apply_learned_mpc.py` 실행 결과에서 확인:

```
📊 Performance Comparison: Original vs Learned

Metric                   Original      Learned    Winner
─────────────────────────────────────────────────────────
RMSE (전체)               0.0234        0.0189   ✅ Learned
Rise Time                 2.456         2.123   ✅ Learned
Overshoot                12.3 %         8.7 %   ✅ Learned
...
```

### 시각화 분석

`ioc_results.png`에서 확인:

1. **Position Trajectory**: 시연 궤적 샘플
2. **Velocity Trajectory**: 속도 프로파일
3. **Control Input**: 토크 입력
4. **‖∇L‖ Comparison**: 초기 vs 학습된 파라미터의 기울기 norm

## 고급 설정

### 최적화 옵션 수정

`inverse_optimal_control.py`에서:

```python
result = minimize(
    objective,
    theta_init,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={
        'maxiter': 100,      # 반복 횟수
        'ftol': 1e-6,        # 함수값 허용 오차
        'disp': True         # 진행상황 출력
    }
)
```

### 다른 최적화 방법 시도

```python
# Trust-Region 방법
from scipy.optimize import minimize

result = minimize(
    objective,
    theta_init,
    method='trust-constr',
    ...
)

# Global optimization (느리지만 더 강건)
from scipy.optimize import differential_evolution

bounds_list = [(1e-3, 1e5)] * 4
result = differential_evolution(
    objective,
    bounds_list,
    ...
)
```

### 파라미터 범위 조정

합리적인 탐색 범위 설정:

```python
bounds = [
    (100, 5000),    # Q_pos
    (10, 200),      # Q_vel
    (1e-4, 1.0),    # R_tau
    (500, 10000)    # Q_terminal
]
```

## 문제 해결

### Q1: "Optimization failed" 오류

**원인:** 초기값이 나쁘거나 제약조건이 너무 엄격함

**해결:**
1. `theta_init` 값 조정
2. `ftol` 값 완화 (1e-6 → 1e-4)
3. `maxiter` 증가 (100 → 200)

### Q2: 학습된 파라미터가 이상함 (너무 크거나 작음)

**원인:** 정규화 제약 부족

**해결:**
```python
# 정규화 강도 조정
constraints = [
    {'type': 'eq', 'fun': lambda θ: np.sum(θ) - 2550.0},
    {'type': 'ineq', 'fun': lambda θ: θ[0] - 100},  # Q_pos 하한
    {'type': 'ineq', 'fun': lambda θ: 5000 - θ[0]}  # Q_pos 상한
]
```

### Q3: ‖∇L‖²이 줄어들지 않음

**원인:** 
1. 시연 데이터가 실제로 최적이 아님
2. 비용함수 형태가 적절하지 않음

**해결:**
1. 더 좋은 시연 수집
2. 비용함수 파라미터화 변경

### Q4: 학습된 가중치의 성능이 더 나쁨

**원인:**
1. Overfitting (시연에만 최적화)
2. 시연 품질이 낮음

**해결:**
1. 더 다양한 시연 수집
2. L2 정규화 추가:
```python
def objective(theta):
    grad_norm = ...
    regularization = 0.01 * np.sum((theta - theta_init)**2)
    return grad_norm + regularization
```

## 확장 가능성

### 1. 다중 목표 최적화

여러 성능 지표를 동시에 고려:

```python
def multi_objective(theta):
    tracking_error = ...
    control_effort = ...
    return w1 * tracking_error + w2 * control_effort
```

### 2. 온라인 학습

시뮬레이션 중에 실시간으로 파라미터 업데이트:

```python
# 각 에피소드마다
theta = ioc.learn_cost_weights([latest_demo], theta_init=theta_current)
apply_learned_weights_to_mpc(controller, theta)
```

### 3. 제약조건 학습

비용함수뿐만 아니라 제약조건도 학습:

```python
# Lagrange 승수 분석
lambda_active = find_active_constraints(demos)
learned_constraints = construct_constraints(lambda_active)
```

## 참고문헌

### 논문

1. **Englert et al. (2017)**: "Inverse KKT: Learning Cost Functions of Manipulation Tasks from Demonstrations"
2. **Menner et al. (2019)**: "Constrained Inverse Optimal Control with Application to a Human Manipulation Task"
3. **Aswani et al. (2018)**: "Inverse Optimization with Noisy Data"

### 이론

- **KKT Conditions**: Karush-Kuhn-Tucker 최적성 조건
- **Convex Optimization**: Boyd & Vandenberghe
- **Inverse Reinforcement Learning**: Abbeel & Ng (2004)

## 라이센스

MIT License

## 문의

이슈나 질문이 있으시면 GitHub Issues에 남겨주세요.

---

**마지막 업데이트:** 2026-02-09
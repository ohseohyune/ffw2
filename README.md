# MPC + Residual Learning for Robot Control

Model Predictive Control (MPC)와 Neural Network 기반 Residual Learning을 결합한 로봇 제어 시스템입니다.


## 🎯 프로젝트 개요

이 프로젝트는 MPC의 모델링 오차를 Neural Network로 보정하여 로봇 제어 성능을 향상시키는 시스템입니다.

### 핵심 아이디어

```
MPC (Model-based) + NN (Data-driven) = 최적의 제어 성능

τ_total = τ_mpc + Δτ_learned
          ^^^^^^   ^^^^^^^^^^^
          물리 기반  학습 기반 보정
```

### 3단계 파이프라인

```
Phase 1: Data Collection (dataGet/)
    └─ MPC 제어 + Residual 토크 수집
        └─ delta_tau_dataset.npz

Phase 2: Neural Network Training (TrainNN/)
    └─ Residual 예측 모델 학습
        └─ residual_nn.pt

Phase 3: Deployment (ApplyNNtoMPC/)
    └─ MPC + NN 결합 제어
        └─ 개선된 추종 성능
```

---

## ✨ 주요 특징

### 1. 고성능 MPC 구현
- ✅ **RK4 Integration**: 4차 정확도 예측
- ✅ **Dynamics Caching**: 계산 효율성 극대화
- ✅ **Velocity Reference Tracking**: 위상 지연 감소
- ✅ **비동기 실행**: 시뮬레이션과 독립적인 MPC 스레드

### 2. Residual Learning
- ✅ **모델 오차 보정**: 마찰, 비선형성, 외란 등
- ✅ **안전한 학습**: Clamping + Gain 조절

### 3. 데이터 파이프라인
- ✅ **자동 데이터 수집**: 시뮬레이션 중 자동 저장
- ✅ **성능 분석**: RMSE, 오버슈트, 정착 시간 등
- ✅ **시각화**: 6가지 토크 비교 그래프

---

## 🏗️ 시스템 구조

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   MuJoCo Simulation                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │                                                   │  │
│  │  Reference Generator ──→ MPC Controller (Thread)  │  │
│  │         ↓                        ↓                │  │
│  │  Robot State ──→ Residual NN ──→ τ_total          │  │
│  │                                   ↓               │  │
│  │                            Apply Torque           │  │
│  │                                   ↓               │  │
│  │                          Update Physics           │  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 모듈 구성

| 모듈 | 역할 | 주요 파일 |
|------|------|----------|
| **dataGet/** | 데이터 수집 | `main.py`, `mpc_controller.py` |
| **TrainNN/** | NN 학습 | `train.py` |
| **ApplyNNtoMPC/** | 배포 | `applyNNtoMPC.py` |
| **eval/** | 성능 평가 | `evaluation.py` |

---

## 🚀 사용 방법

### Quick Start

```bash
# 1. 데이터 수집 (Phase 1)
cd /path/to/project
python -m dataGet.main

# 2. NN 학습 (Phase 2)
python -m TrainNN.train

# 3. 적용 (Phase 3)
python -m ApplyNNtoMPC.applyNNtoMPC

#4. 평가 (Phase 4)
python -m eval.evaluation
```

### 단계별 상세 설명

#### Phase 1: 데이터 수집

```bash
python -m dataGet.main
```

**출력**:
- `delta_tau_dataset.npz`: 학습 데이터셋
  - `q`: 관절 위치 [N, 1]
  - `qdot`: 관절 속도 [N, 1]
  - `tau_mpc`: MPC 토크 [N, 1]
  - `delta_tau`: Residual 토크 [N, 1] ← **Label**

**시각화**:
- 관절 추종 성능
- 추종 오차
- 적용 토크
- Residual 토크 분포

#### Phase 2: NN 학습

```bash
python -m TrainNN.train
```

**학습 설정** (`train.py` 수정):
```python
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3
DELTA_TAU_MAX = 50.0  # Residual 최대값 [Nm]
```

**출력**:
- `residual_nn.pt`: 학습된 모델
- Training/Validation loss 그래프
- 검증셋 오차 통계

#### Phase 3: 배포

```bash
python -m ApplyNNtoMPC.applyNNtoMPC
```

**설정** (`applyNNtoMPC.py` 수정):
```python
residual_comp = ResidualCompensator(
    model_path='residual_nn.pt',
    alpha=2.0,           # Residual 게인 (조절 가능)
    delta_tau_max=50.0   # Clamp 한계
)
```

**출력**:
- 실시간 제어 시각화
- 성능 비교 (MPC only vs MPC+NN)
- 6가지 토크 비교 그래프

---

## 📁 프로젝트 구조

```
ffw/code/
├── dataGet/                    # Phase 1: 데이터 수집
│   ├── __init__.py            # 공개 API
│   ├── config.py              # 전체 설정
│   ├── main.py                # 메인 실행 스크립트
│   ├── mpc_controller.py      # MPC 컨트롤러 (RK4, caching)
│   ├── async_utils.py         # 비동기 MPC 관리
│   ├── residual_calculator.py # Residual 토크 계산
│   ├── robot_setup.py         # 로봇 인터페이스
│   ├── trajectory.py          # 참조 궤적 생성
│   ├── data_logger.py         # 데이터 로깅
│   ├── visualization.py       # 시각화 도구
│   └── neural_network.py      # NN 모델 정의
│
├── TrainNN/                    # Phase 2: NN 학습
│   ├── train.py               # 학습 스크립트
│   └── residual_nn.pt         # 학습된 모델 (출력)
│
├── ApplyNNtoMPC/               # Phase 3: 배포
│   └── applyNNtoMPC.py        # MPC+NN 배포 스크립트
│
├── eval/                       # 성능 평가
│   └── evaluation.py          # 정량적 평가 도구
│
└── README.md                   # 본 문서
```

---

## ⚙️ 설정 파일

모든 설정은 `dataGet/config.py`에 집중되어 있습니다.

### 주요 설정 클래스

#### 1. SimulationConfig
```python
class SimulationConfig:
    SIM_DT = 0.005           # 시뮬레이션 타임스텝 [s] (200 Hz)
    MPC_RATE_HZ = 80.0       # MPC 실행 빈도 [Hz]
    SIM_DURATION = 10.0      # 총 시뮬레이션 시간 [s]
    REALTIME_FACTOR = 1.0    # 실시간 배속
```

#### 2. MPCConfig
```python
class MPCConfig:
    HORIZON = 20             # 예측 수평선 (steps)
    MAX_ITER = 50            # SLSQP 최대 반복
    FTOL = 1e-5             # 수렴 허용 오차
```

#### 3. CostWeights
```python
class CostWeights:
    Q_POS = np.eye(1) * 2000.0       # 위치 추종 가중치
    Q_VEL_REF = np.eye(1) * 10.0     # 속도 참조 가중치
    R_TAU = np.eye(1) * 0.001        # 제어 노력 가중치
    Q_TERMINAL = np.eye(1) * 2500.0  # 종단 비용
```

#### 4. TrajectoryConfig
```python
class TrajectoryConfig:
    SHOULDER_START = 0.0     # 시작 각도 [rad]
    SHOULDER_TARGET = -1.5   # 목표 각도 [rad]
    T_RAISE = 5.0           # 팔 올리기 시간 [s]
    T_WAIT = 0.5            # 대기 시간 [s]
    T_HOLD = 1.0            # 유지 시간 [s]
```

#### 5. ResidualNNConfig
```python
class ResidualNNConfig:
    DELTA_TAU_MAX = 50.0     # Residual 최대값 [Nm]
    INPUT_DIM = 3            # [q, qdot, tau_mpc]
    HIDDEN_DIM = 64          # 은닉층 차원
    OUTPUT_DIM = 1           # [delta_tau]
```

---

## 📊 결과 예시

### 성능 비교 (1 DOF Shoulder Joint)

```
┌──────────────────────────────────────────────────────────┐
│  📊 Performance Comparison: MPC only vs MPC + NN         │
├──────────────────────────────────────────────────────────┤
│  Metric                    MPC only      MPC + NN        │
├──────────────────────────────────────────────────────────┤
│  RMSE (전체)              0.0234 rad    0.0089 rad  ◀ NN │
│  RMSE (전이구간)          0.0312 rad    0.0124 rad  ◀ NN │
│  RMSE (정지구간)          0.0087 rad    0.0032 rad  ◀ NN │
│  Max Error                0.0891 rad    0.0234 rad  ◀ NN │
│  Rise Time (90%)          2.34 s        2.12 s      ◀ NN │
│  Settling Time (±2%)      3.21 s        2.87 s      ◀ NN │
│  Overshoot                12.3 %        4.7 %       ◀ NN │
│  Mean |tau|               15.4 Nm       16.2 Nm     ◀ MPC│
│  Peak |tau|               45.2 Nm       48.1 Nm     ◀ MPC│
└──────────────────────────────────────────────────────────┘

성능 개선:
  - RMSE: 62% 감소 ⬇
  - 오버슈트: 62% 감소 ⬇
  - 정착 시간: 11% 단축 ⬇
  - 제어 노력: 5% 증가 (허용 범위) ⬆
```

## 🔬 고급 사용법

### 1. Hyperparameter Tuning

#### MPC 파라미터 튜닝
```python
# config.py 수정
class MPCConfig:
    HORIZON = 20        # 길수록 정확, 느림
    
class CostWeights:
    Q_POS = np.eye(1) * 5000.0    # 높을수록 빠른 추종
    R_TAU = np.eye(1) * 0.0001    # 낮을수록 큰 토크 허용
```

#### NN 파라미터 튜닝
```python
# train.py 수정
BATCH_SIZE = 256      # 큰 배치: 안정, 느림
LR = 5e-4            # 낮은 lr: 안정, 느림
EPOCHS = 300         # 많은 epoch: 정확, 느림
```

### 2. 다른 궤적 실험

```python
# trajectory.py 수정
class TrajectoryConfig:
    # 빠른 운동
    T_RAISE = 2.0  # 5.0 → 2.0
    
    # 큰 각도
    SHOULDER_TARGET = -2.0  # -1.5 → -2.0
```

### 3. 다중 관절 확장

```python
# config.py에서:
class RobotConfig:
    JOINT_NAMES = [
        "arm_r_joint1",
        "arm_r_joint3",  # 추가!
        "arm_r_joint7"   # 추가!
    ]

# 모든 eye(1) → eye(3)으로 변경
class CostWeights:
    Q_POS = np.eye(3) * 2000.0  # ← 1 → 3
```


## 📄 라이센스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Authors

**Your Name**
- GitHub: [ohseohyune]
- Email: ohseohyun0531@naver.com

---

## 🙏 감사의 말

Danke!


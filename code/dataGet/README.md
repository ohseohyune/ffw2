# MPC + Residual Torque Dataset Generation

로봇 팔 제어를 위한 MPC(Model Predictive Control)와 Residual Torque 데이터셋 생성 프로젝트입니다.

## 📁 프로젝트 구조

```
mpc_project/
├── config.py                  # 모든 설정 파라미터
├── trajectory.py              # 참조 궤적 생성
├── neural_network.py          # Residual Torque NN 모델
├── mpc_controller.py          # MPC 컨트롤러
├── async_utils.py             # 비동기 MPC 유틸리티
├── data_logger.py             # 데이터 수집 및 로깅
├── residual_calculator.py     # Residual 토크 계산
├── robot_setup.py             # 로봇 인터페이스
├── visualization.py           # 시각화 도구
├── main.py                    # 메인 실행 파일
└── README.md                  # 이 파일
```

## 🎯 프로젝트 흐름

```
1. main.py (MPC 시뮬레이션 + 데이터수집)
   ↓
   delta_tau_dataset.npz 생성
   ↓
2. train_residual_nn.py (신경망 학습)
   ↓
   학습된 모델 생성
   ↓
3. apply_nn.py (학습된 모델 적용)
```

## 🚀 사용 방법

### 1. 데이터셋 생성 (이 프로젝트)

```bash
python main.py
```

**출력:**
- `delta_tau_dataset.npz`: 학습용 데이터셋
  - `q`: 관절 위치 (N, 3)
  - `qdot`: 관절 속도 (N, 3)
  - `tau_mpc`: MPC 토크 (N, 3)
  - `delta_tau`: 잔여 토크 (N, 3) ← 학습 라벨

**시각화:**
- 관절 추적 성능 그래프
- 추적 오차 그래프
- 적용된 토크 그래프
- Residual 토크 크기 막대 그래프

### 2. 신경망 학습 (별도 파일)

```bash
python train_residual_nn.py
```

### 3. 학습된 모델 적용 (별도 파일)

```bash
python apply_nn.py
```

## ⚙️ 주요 설정 파라미터

모든 설정은 `config.py`에서 수정할 수 있습니다.

### 1. 시뮬레이션 설정 (`SimulationConfig`)

```python
SIM_DT = 0.005              # 시뮬레이션 타임스텝 (200 Hz)
MPC_RATE_HZ = 100.0         # MPC 실행 주파수
SIM_DURATION = 5.5          # 전체 시뮬레이션 시간
REALTIME_FACTOR = 0.5       # 실시간 속도 제한 (50%)
```

### 2. MPC 설정 (`MPCConfig`)

```python
HORIZON = 10                # MPC 예측 수평선
MAX_ITER = 50              # 최적화 최대 반복 횟수
FTOL = 1e-5                # 수렴 허용 오차
```

### 3. 비용함수 가중치 (`CostWeights`)

```python
Q_POS = np.eye(3) * 500.0      # 위치 오차 가중치
R_TAU = np.eye(3) * 0.01       # 토크 입력 가중치
Q_TERMINAL = np.eye(3) * 800.0 # 종단 위치 오차 가중치
```

### 4. 토크 제약 (`TorqueLimits`)

```python
TAU_MAX = 250.0             # 최대 토크 [Nm]
TAU_MIN = -250.0            # 최소 토크 [Nm]
```

### 5. Residual NN 설정 (`ResidualNNConfig`)

```python
DELTA_TAU_MAX = 50.0        # Residual 토크 최대 크기 [Nm]
INPUT_DIM = 9               # 입력 차원
HIDDEN_DIM = 64             # 은닉층 차원
OUTPUT_DIM = 3              # 출력 차원
```

## 📊 데이터셋 구조

**delta_tau_dataset.npz** 파일 내용:

| 변수 | 형태 | 설명 |
|------|------|------|
| `q` | (N, 3) | 관절 위치 [rad] |
| `qdot` | (N, 3) | 관절 속도 [rad/s] |
| `tau_mpc` | (N, 3) | MPC 토크 [Nm] |
| `delta_tau` | (N, 3) | 잔여 토크 (실제 - MPC) [Nm] |

N = 시뮬레이션 스텝 수 ≈ 1100

## 🎮 제어 대상 관절

1. **Shoulder (arm_r_joint1)**: 어깨 관절
2. **Upper Arm (arm_r_joint3)**: 상완 관절
3. **Wrist (arm_r_joint7)**: 손목 관절

## 📈 참조 궤적

### Phase 1: 팔 올리기 (0 ~ 2초)
- 어깨를 0°에서 -2.8 rad로 부드럽게 이동

### Phase 2: 대기 (2 ~ 2.5초)
- 위치 유지

### Phase 3: 손 흔들기 (2.5 ~ 4.5초)
- 손목: 0.3 rad 진폭으로 0.5 Hz 사인파
- 상완: 0.5 rad 진폭으로 0.5 Hz 사인파 (위상 지연 포함)

### Phase 4: 종료 대기 (4.5 ~ 5.5초)
- 최종 위치 유지

## 🔧 커스터마이징

### 1. 다른 로봇 사용

`config.py`의 `RobotConfig`에서 관절 이름 변경:

```python
class RobotConfig:
    SHOULDER_JOINT_NAME = "your_shoulder_joint"
    UPPERARM_JOINT_NAME = "your_upperarm_joint"
    WRIST_JOINT_NAME = "your_wrist_joint"
    
    MOTOR_NAMES = [
        "motor_shoulder",
        "motor_upperarm",
        "motor_wrist"
    ]
```

### 2. 궤적 변경

`config.py`의 `TrajectoryConfig`에서 파라미터 수정:

```python
class TrajectoryConfig:
    SHOULDER_TARGET = -2.8      # 목표 각도
    T_RAISE = 2.0               # 올리기 시간
    WRIST_AMPLITUDE = 0.3       # 흔들기 진폭
    WAVE_FREQUENCY = 0.5        # 흔들기 주파수
    # ...
```

### 3. MPC 성능 튜닝

`config.py`에서 가중치 조정:

```python
# 위치 추적을 더 중요하게
Q_POS = np.eye(3) * 1000.0

# 토크 사용을 줄이고 싶으면
R_TAU = np.eye(3) * 0.1

# 예측 수평선 늘리기 (더 먼 미래를 고려)
HORIZON = 15
```

## 🏗️ 아키텍처

### Async MPC 구조

```
[Simulation Thread]           [MPC Thread]
    (200 Hz)                     (100 Hz)
        |                            |
        |-- push input -->  SharedMPCInput
        |                            |
        |                      [MPC Solve]
        |                            |
        |<-- read torque -- SharedTorqueBuffer
        |                            |
    [Apply & Step]              [Loop]
```

- **Simulation**: 200 Hz로 빠르게 실행
- **MPC**: 100 Hz로 독립적으로 최적화
- **Thread-safe**: Lock으로 보호된 버퍼 통신

## 📝 주요 클래스

### TorqueMPC
- MPC 컨트롤러
- SLSQP 최적화로 토크 시퀀스 계산
- 비용함수: 위치 추적 + 토크 최소화 + 종단 비용

### ResidualCalculator
- 실제 필요 토크 vs MPC 토크 차이 계산
- MuJoCo dynamics를 사용한 정확한 계산

### DatasetCollector
- 학습 데이터 수집
- 통계 분석 기능

### TrackingLogger
- 제어 성능 추적
- RMSE, 최대 오차 등 계산

## 🎯 가장 중요한 파라미터 Top 5

1. **HORIZON**: MPC 예측 길이 (성능 vs 계산 속도)
2. **Q_POS / R_TAU / Q_TERMINAL**: 비용함수 가중치 (제어 성향)
3. **MPC_RATE_HZ**: MPC 실행 주파수 (반응 속도)
4. **DELTA_TAU_MAX**: Residual NN 출력 범위
5. **참조 궤적**: 데이터 다양성과 품질

## ⚠️ 주의사항

1. **XML 경로**: `PathConfig`에서 MuJoCo XML 파일 경로가 올바른지 확인
2. **관절 이름**: 로봇 모델의 실제 관절 이름과 일치해야 함
3. **토크 제한**: 실제 로봇의 토크 제한에 맞게 설정
4. **실시간 제약**: MPC 최적화 시간이 제어 주기보다 짧아야 함

## 📚 참고 자료

- MuJoCo Documentation: https://mujoco.readthedocs.io/
- Model Predictive Control: https://en.wikipedia.org/wiki/Model_predictive_control
- PyTorch: https://pytorch.org/

## 🐛 문제 해결

### MPC가 너무 느림
```python
# config.py에서 조정
HORIZON = 5          # 줄이기
MAX_ITER = 30        # 줄이기
MPC_RATE_HZ = 50.0   # 낮추기
```

### 추적 성능이 나쁨
```python
# 가중치 증가
Q_POS = np.eye(3) * 1000.0
Q_TERMINAL = np.eye(3) * 1500.0

# 또는 MPC 주파수 증가
MPC_RATE_HZ = 200.0
```

### 토크가 진동함
```python
# 토크 비용 증가
R_TAU = np.eye(3) * 0.1

# 또는 예측 수평선 증가
HORIZON = 15
```

<!-- ## 📞 연락처

문의사항이 있으시면 이슈를 등록해주세요.

---

**License**: 
**Author**: Your Name
**Date**: 2026-01-28 -->
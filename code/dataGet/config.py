# """
# Configuration file for MPC + Residual Torque Dataset Generation

# 모든 설정 가능한 파라미터를 여기에 모았습니다.
# """

# import numpy as np
# import os


# class SimulationConfig:
#     """시뮬레이션 시간/주기 설정"""
    
#     # 시뮬레이션 기본 타임스텝 (200 Hz)
#     SIM_DT = 0.005
    
#     # MPC 실행 주파수 (100 Hz)
#     MPC_RATE_HZ = 60.0
    
#     # 전체 시뮬레이션 시간
#     SIM_DURATION = 10  # T_raise + T_wait + T_wave + 1.0
    
#     # 시뮬레이션 속도 제한 (실시간의 50%)
#     REALTIME_FACTOR = 1.0


# class MPCConfig:
#     """MPC 구조 및 최적화 설정"""
    
#     # MPC 예측 수평선
#     HORIZON = 7
    
#     # SLSQP 옵티마이저 설정
#     MAX_ITER = 50
#     FTOL = 1e-5


# class CostWeights:
#     """비용함수 가중치 (제어 성향)"""
    
#     # 위치 오차 가중치
#     Q_POS = np.eye(3) * 2000.0
    
#     # 토크 입력 가중치 (작을수록 큰 토크 허용)
#     R_TAU = np.eye(3) * 0.001
    
#     # 종단 위치 오차 가중치
#     Q_TERMINAL = np.eye(3) * 2500.0

#     # 속도 가중치 (damping 효과)
#     Q_VEL = np.eye(3) * 50.0

#     # 종단 속도 가중치
#     Q_VEL_TERMINAL = np.eye(3) * 50.0

#     # 속도 참조 추적 가중치
#     Q_VEL_REF = np.eye(3) * 10.0


# class TorqueLimits:
#     """토크 및 물리 제약"""
    
#     # 최대 토크 [Nm]
#     TAU_MAX = 100.0
    
#     # 최소 토크 [Nm]
#     TAU_MIN = -100.0


# class ResidualNNConfig:
#     """Residual NN 설계와 스케일"""
    
#     # Residual torque의 최대 크기 [Nm]
#     DELTA_TAU_MAX = 50.0
    
#     # 네트워크 구조
#     INPUT_DIM = 9   # q(3) + qdot(3) + tau_mpc(3)
#     HIDDEN_DIM = 64
#     OUTPUT_DIM = 3  # delta_tau(3)


# class RobotConfig:
#     """로봇 관절 설정"""
    
#     # 제어할 관절 이름
#     SHOULDER_JOINT_NAME = "arm_r_joint1"
#     # UPPERARM_JOINT_NAME = "arm_r_joint3"
#     # WRIST_JOINT_NAME = "arm_r_joint7"
    
#     # 모터 이름
#     MOTOR_NAMES = [
#         "motor_arm_r_joint1"
#         # "motor_arm_r_joint3", 
#         # "motor_arm_r_joint7"
#     ]


# class TrajectoryConfig:
#     """참조 궤적 파라미터"""
    
#     # Phase 1: 팔 올리기
#     SHOULDER_START = 0.0
#     SHOULDER_TARGET = -1.5
#     T_RAISE = 3.0
    
#     # Phase 2: 대기
#     T_WAIT = 0.5
    
#     # Phase 3: 손 흔들기
#     T_WAVE = 3.0
#     WRIST_CENTER = 0.0
#     WRIST_AMPLITUDE = 0.0
#     WAVE_FREQUENCY = 0.0
    
#     UPPERARM_CENTER = 0.0
#     UPPERARM_AMPLITUDE = 0.2
#     PHASE_DELAY = 0.0


# class PathConfig:
#     """파일 경로 설정"""
    
#     # 절대 경로로 직접 지정 (사용자 환경에 맞게 수정 필요)
#     ROOT_DIR = "/home/seohy/colcon_ws/src/olaf/ffw"
#     CODE_DIR = os.path.join(ROOT_DIR, "code")
#     MODEL_DIR = os.path.join(ROOT_DIR, "model")
#     XML_PATH = os.path.join(MODEL_DIR, "scene_ffw_sg2.xml")
    
#     @staticmethod
#     def get_paths():
#         """프로젝트 경로 반환"""
#         return {
#             'code_dir': PathConfig.CODE_DIR,
#             'root_dir': PathConfig.ROOT_DIR,
#             'model_dir': PathConfig.MODEL_DIR,
#             'xml_path': PathConfig.XML_PATH
#         }
    
#     # 데이터셋 저장 경로
#     DATASET_PATH = "/home/seohy/colcon_ws/src/olaf/ffw/code/dataGet/delta_tau_dataset.npz"


# class DatasetConfig:
#     """데이터셋 생성 관련 설정"""
    
#     # MPC 입력 푸시 간격 (steps)
#     @staticmethod
#     def get_push_interval_steps():
#         return int((1.0 / SimulationConfig.MPC_RATE_HZ) / SimulationConfig.SIM_DT)
    
#     # 전체 스텝 수
#     @staticmethod
#     def get_total_steps():
#         return int(SimulationConfig.SIM_DURATION / SimulationConfig.SIM_DT)

"""
Configuration file for MPC + Residual Torque Dataset Generation

모든 설정 가능한 파라미터를 여기에 모았습니다.
"""

import numpy as np
import os


class SimulationConfig:
    """시뮬레이션 시간/주기 설정"""

    # 시뮬레이션 기본 타임스텝 (200 Hz)
    SIM_DT = 0.005

    # MPC 실행 주파수 (100 Hz)
    MPC_RATE_HZ = 200.0

    # 전체 시뮬레이션 시간
    SIM_DURATION = 10  # T_raise + T_wait + T_wave + 1.0

    # 시뮬레이션 속도 제한 (실시간의 50%)
    REALTIME_FACTOR = 1.0


class MPCConfig:
    """MPC 구조 및 최적화 설정"""

    # MPC 예측 수평선
    HORIZON = 20

    # SLSQP 옵티마이저 설정
    MAX_ITER = 50
    FTOL = 1e-5


class CostWeights:
    """비용함수 가중치 (제어 성향) — 1DOF (shoulder)"""

    # 위치 오차 가중치
    Q_POS = np.eye(1) * 2000.0                  # 수정! eye(3) → eye(1)

    # 토크 입력 가중치 (작을수록 큰 토크 허용)
    R_TAU = np.eye(1) * 0.001                   # 수정! eye(3) → eye(1)

    # 종단 위치 오차 가중치
    Q_TERMINAL = np.eye(1) * 2500.0             # 수정! eye(3) → eye(1)

    # 속도 가중치 (damping 효과)
    Q_VEL = np.eye(1) * 50.0                    # 수정! eye(3) → eye(1)

    # 종단 속도 가중치
    Q_VEL_TERMINAL = np.eye(1) * 50.0           # 수정! eye(3) → eye(1)

    # 속도 참조 추적 가중치
    Q_VEL_REF = np.eye(1) * 10.0                # 수정! eye(3) → eye(1)


class TorqueLimits:
    """토크 및 물리 제약"""

    # 최대 토크 [Nm]
    TAU_MAX = 100.0

    # 최소 토크 [Nm]
    TAU_MIN = -100.0


class ResidualNNConfig:
    """Residual NN 설계와 스케일 — 1DOF (shoulder)"""

    # Residual torque의 최대 크기 [Nm]
    DELTA_TAU_MAX = 50.0

    # 네트워크 구조
    INPUT_DIM = 3    # 수정! q(1) + qdot(1) + tau_mpc(1)
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1   # 수정! delta_tau(1)


class RobotConfig:
    """로봇 관절 설정 — 1DOF (shoulder)"""

    # 제어할 관절 이름
    SHOULDER_JOINT_NAME = "arm_r_joint1"
    # 수정! upperarm, wrist 관련 제거

    # 모터 이름
    MOTOR_NAMES = [
        "motor_arm_r_joint1"
        # 수정! upperarm, wrist 모터 제거
    ]


class TrajectoryConfig:
    """참조 궤적 파라미터 — 1DOF (shoulder)"""

    # Phase 1: 팔 올리기
    SHOULDER_START = 0.0
    SHOULDER_TARGET = -1.5
    T_RAISE = 5.0

    # Phase 2: 대기
    T_WAIT = 0.5

    # Phase 3: 유지 
    T_HOLD = 1.0                                



class PathConfig:
    """파일 경로 설정"""

    # 절대 경로로 직접 지정 (사용자 환경에 맞게 수정 필요)
    ROOT_DIR = "/home/seohy/colcon_ws/src/ffw2"
    CODE_DIR = os.path.join(ROOT_DIR, "code")
    MODEL_DIR = os.path.join(ROOT_DIR, "model")
    XML_PATH = os.path.join(MODEL_DIR, "scene_ffw_sg2.xml")

    @staticmethod
    def get_paths():
        """프로젝트 경로 반환"""
        return {
            'code_dir': PathConfig.CODE_DIR,
            'root_dir': PathConfig.ROOT_DIR,
            'model_dir': PathConfig.MODEL_DIR,
            'xml_path': PathConfig.XML_PATH
        }

    # 데이터셋 저장 경로
    DATASET_PATH = "/home/seohy/colcon_ws/src/ffw2/code/dataGet/delta_tau_dataset.npz"


class DatasetConfig:
    """데이터셋 생성 관련 설정"""

    # MPC 입력 푸시 간격 (steps)
    @staticmethod
    def get_push_interval_steps():
        return int((1.0 / SimulationConfig.MPC_RATE_HZ) / SimulationConfig.SIM_DT)

    # 전체 스텝 수
    @staticmethod
    def get_total_steps():
        return int(SimulationConfig.SIM_DURATION / SimulationConfig.SIM_DT)
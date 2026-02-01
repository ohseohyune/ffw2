# """
# Reference Trajectory Generation

# 팔 흔들기 동작을 위한 참조 궤적을 생성합니다.
# """

# import numpy as np
# from .config import TrajectoryConfig


# def generate_reference_trajectory(t, shoulder_qpos_id, upperarm_qpos_id, wrist_qpos_id):
#     """
#     Generate reference trajectory for waving motion
    
#     Args:
#         t: Current time [s]
#         shoulder_qpos_id: qpos index for shoulder joint
#         upperarm_qpos_id: qpos index for upper arm joint
#         wrist_qpos_id: qpos index for wrist joint
    
#     Returns:
#         q_ref_dict: Dictionary mapping qpos_id -> desired angle [rad]
#     """
#     cfg = TrajectoryConfig
    
#     # Phase 1: 팔 올리기 (Raising arm)
#     if t <= cfg.T_RAISE:
#         s = np.clip(t / cfg.T_RAISE, 0.0, 1.0)
#         shoulder_des = (1 - s) * cfg.SHOULDER_START + s * cfg.SHOULDER_TARGET
#         upperarm_des = cfg.UPPERARM_CENTER
#         wrist_des = cfg.WRIST_CENTER
        
#     # Phase 2: 대기 (Waiting)
#     elif t <= cfg.T_RAISE + cfg.T_WAIT:
#         shoulder_des = cfg.SHOULDER_TARGET
#         upperarm_des = cfg.UPPERARM_CENTER
#         wrist_des = cfg.WRIST_CENTER
        
#     # Phase 3: 손 흔들기 (Waving)
#     elif t <= cfg.T_RAISE + cfg.T_WAIT + cfg.T_WAVE:
#         shoulder_des = cfg.SHOULDER_TARGET
        
#         t_wave = t - (cfg.T_RAISE + cfg.T_WAIT)
#         wrist_des = cfg.WRIST_CENTER + cfg.WRIST_AMPLITUDE * np.sin(
#             2 * np.pi * cfg.WAVE_FREQUENCY * t_wave
#         )
        
#         t_wave_delayed = t_wave - cfg.PHASE_DELAY
#         if t_wave_delayed > 0:
#             upperarm_des = cfg.UPPERARM_CENTER + cfg.UPPERARM_AMPLITUDE * np.sin(
#                 2 * np.pi * cfg.WAVE_FREQUENCY * t_wave_delayed
#             )
#         else:
#             upperarm_des = cfg.UPPERARM_CENTER
            
#     # Phase 4: 종료 대기 (Final hold)
#     else:
#         shoulder_des = cfg.SHOULDER_TARGET
#         upperarm_des = cfg.UPPERARM_CENTER
#         wrist_des = cfg.WRIST_CENTER

#     # Return as dictionary for easy mapping
#     q_ref_dict = {
#         shoulder_qpos_id: shoulder_des,
#         upperarm_qpos_id: upperarm_des,
#         wrist_qpos_id: wrist_des
#     }
    
#     return q_ref_dict


# def get_trajectory_phases():
#     """
#     Get trajectory phase timing information
    
#     Returns:
#         dict: Phase names and their end times
#     """
#     cfg = TrajectoryConfig
    
#     return {
#         'raise': cfg.T_RAISE,
#         'wait': cfg.T_RAISE + cfg.T_WAIT,
#         'wave': cfg.T_RAISE + cfg.T_WAIT + cfg.T_WAVE
#     }

"""
Reference Trajectory Generation

팔 올리기 동작을 위한 참조 궤적을 생성합니다. (1DOF: shoulder)
"""

import numpy as np
from .config import TrajectoryConfig


def generate_reference_trajectory(t, shoulder_qpos_id):  # 수정! upperarm/wrist 인자 제거
    """
    Generate reference trajectory for shoulder raise motion.
    1DOF (shoulder) 만 대상.

    Args:
        t: Current time [s]
        shoulder_qpos_id: qpos index for shoulder joint

    Returns:
        q_ref_dict: {shoulder_qpos_id: desired_angle [rad]}
    """
    cfg = TrajectoryConfig

    # Phase 1: 팔 올리기 (Raising arm)
    if t <= cfg.T_RAISE:
        s = np.clip(t / cfg.T_RAISE, 0.0, 1.0)
        shoulder_des = (1 - s) * cfg.SHOULDER_START + s * cfg.SHOULDER_TARGET

    # Phase 2: 대기 (Waiting)
    elif t <= cfg.T_RAISE + cfg.T_WAIT:
        shoulder_des = cfg.SHOULDER_TARGET

    # Phase 3: 유지 (Holding) — 수정! wave 로직 제거, 단순 유지
    elif t <= cfg.T_RAISE + cfg.T_WAIT + cfg.T_HOLD:
        shoulder_des = cfg.SHOULDER_TARGET

    # Phase 4: 종료 대기 (Final hold)
    else:
        shoulder_des = cfg.SHOULDER_TARGET

    # 수정! shoulder만 반환
    q_ref_dict = {
        shoulder_qpos_id: shoulder_des
    }

    return q_ref_dict


def get_trajectory_phases():
    """
    Get trajectory phase timing information

    Returns:
        dict: Phase names and their end times
    """
    cfg = TrajectoryConfig

    return {
        'raise': cfg.T_RAISE,
        'wait': cfg.T_RAISE + cfg.T_WAIT,
        'hold': cfg.T_RAISE + cfg.T_WAIT + cfg.T_HOLD  # 수정! 'wave' → 'hold'
    }
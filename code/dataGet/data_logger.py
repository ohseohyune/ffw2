# """
# Data Logger for MPC Simulation

# 시뮬레이션 중 데이터를 수집하고 저장합니다.
# """

# import numpy as np
# from .config import PathConfig


# class DatasetCollector:
#     """
#     Residual Torque Dataset Collector
    
#     MPC 시뮬레이션 중 상태, 토크, residual 토크를 수집합니다.
#     """
    
#     def __init__(self):
#         """Initialize empty data lists"""
#         self.dataset_q = []
#         self.dataset_qdot = []
#         self.dataset_tau = []
#         self.dataset_delta_tau = []

#     def add_sample(self, q, qdot, tau_mpc, delta_tau):
#         """
#         Add one sample to dataset
        
#         Args:
#             q: Joint positions for controlled joints
#             qdot: Joint velocities for controlled joints
#             tau_mpc: MPC torque
#             delta_tau: Residual torque (true - MPC)
#         """
#         self.dataset_q.append(q.copy())
#         self.dataset_qdot.append(qdot.copy())
#         self.dataset_tau.append(tau_mpc.copy())
#         self.dataset_delta_tau.append(delta_tau.copy())

#     def save_dataset(self, filepath=None):
#         """
#         Save collected dataset to file
        
#         Args:
#             filepath: Path to save .npz file (uses config default if None)
        
#         Returns:
#             filepath: Path where data was saved
#         """
#         if filepath is None:
#             filepath = PathConfig.DATASET_PATH
        
#         np.savez(
#             filepath,
#             q=np.array(self.dataset_q),
#             qdot=np.array(self.dataset_qdot),
#             tau_mpc=np.array(self.dataset_tau),
#             delta_tau=np.array(self.dataset_delta_tau)
#         )
        
#         print(f"\n✅ Dataset saved to {filepath}")
#         print(f"   Samples: {len(self.dataset_q)}")
        
#         return filepath

#     def get_statistics(self):
#         """
#         Get dataset statistics
        
#         Returns:
#             dict: Statistics including mean, max, and ratio of residual torques
#         """
#         if len(self.dataset_q) == 0:
#             return None
        
#         delta_tau = np.array(self.dataset_delta_tau)
#         tau_mpc = np.array(self.dataset_tau)
        
#         mean_abs = np.mean(np.abs(delta_tau), axis=0)
#         max_abs = np.max(np.abs(delta_tau), axis=0)
#         std_abs = np.std(np.abs(delta_tau), axis=0)
        
#         tau_mpc_mean = np.mean(np.abs(tau_mpc), axis=0)
#         ratio = mean_abs / (tau_mpc_mean + 1e-6)
        
#         return {
#             'mean_abs': mean_abs,
#             'max_abs': max_abs,
#             'std_abs': std_abs,
#             'ratio': ratio,
#             'n_samples': len(self.dataset_q)
#         }


# class TorqueComparisonLogger:
#     """
#     Torque Comparison Logger
    
#     6가지 토크를 모두 기록합니다:
#     1. tau_mpc: MPC가 계산한 토크
#     2. tau_true: 실제 필요한 토크 (from residual calculator)
#     3. tau_residual: tau_true - tau_mpc
#     4. delta_trained: 학습된 NN이 예측한 residual
#     5. tau_total: tau_mpc + delta_trained (실제 적용된 토크)
#     6. tau_final: tau_mpc + alpha * clamp(delta_trained)
#     """
    
#     def __init__(self):
#         """Initialize empty log lists"""
#         self.time_log = []
        
#         # 3개 관절 각각에 대해 6가지 토크 저장
#         # Shape will be: (n_timesteps, 3) for each torque type
#         self.tau_mpc_log = []          # 1. MPC 토크
#         self.tau_true_log = []         # 2. 실제 필요 토크
#         self.tau_residual_log = []     # 3. Residual (true - mpc)
#         self.delta_trained_log = []    # 4. 학습된 NN 예측
#         self.tau_total_log = []        # 5. MPC + delta_trained
#         self.tau_final_log = []        # 6. MPC + alpha*clamp(delta)

#     def add_sample(self, t, tau_mpc, tau_true, delta_trained, tau_final):
#         """
#         Add one sample to log
        
#         Args:
#             t: Time [s]
#             tau_mpc: MPC torque [3]
#             tau_true: True required torque [3]
#             delta_trained: NN predicted residual [3]
#             tau_final: Final applied torque (MPC + alpha*clamp(delta)) [3]
#         """
#         self.time_log.append(t)
#         self.tau_mpc_log.append(tau_mpc.copy())
#         self.tau_true_log.append(tau_true.copy())
#         self.tau_residual_log.append((tau_true - tau_mpc).copy())
#         self.delta_trained_log.append(delta_trained.copy())
#         self.tau_total_log.append((tau_mpc + delta_trained).copy())
#         self.tau_final_log.append(tau_final.copy())

#     def get_arrays(self):
#         """
#         Get all logged data as numpy arrays
        
#         Returns:
#             dict: All logged data as numpy arrays
#         """
#         return {
#             'time': np.array(self.time_log),
#             'tau_mpc': np.array(self.tau_mpc_log),           # (N, 3)
#             'tau_true': np.array(self.tau_true_log),         # (N, 3)
#             'tau_residual': np.array(self.tau_residual_log), # (N, 3)
#             'delta_trained': np.array(self.delta_trained_log), # (N, 3)
#             'tau_total': np.array(self.tau_total_log),       # (N, 3)
#             'tau_final': np.array(self.tau_final_log)        # (N, 3)
#         }


# class TrackingLogger:
#     """
#     Tracking Performance Logger
    
#     제어 성능 추적을 위한 데이터를 수집합니다.
#     """
    
#     def __init__(self):
#         """Initialize empty log lists"""
#         self.time_log = []
#         self.shoulder_ref_log = []
#         self.shoulder_act_log = []
#         self.upperarm_ref_log = []
#         self.upperarm_act_log = []
#         self.wrist_ref_log = []
#         self.wrist_act_log = []
#         self.tau_shoulder_log = []
#         self.tau_upperarm_log = []
#         self.tau_wrist_log = []

#     def add_sample(self, t, q_ref_dict, q_act, tau, 
#                    shoulder_id, upperarm_id, wrist_id):
#         """
#         Add one sample to log
        
#         Args:
#             t: Time [s]
#             q_ref_dict: Reference angles dictionary
#             q_act: Actual joint positions (full)
#             tau: Applied torques for controlled joints
#             shoulder_id: qpos index for shoulder
#             upperarm_id: qpos index for upper arm
#             wrist_id: qpos index for wrist
#         """
#         self.time_log.append(t)
#         self.shoulder_ref_log.append(q_ref_dict[shoulder_id])
#         self.shoulder_act_log.append(q_act[shoulder_id])
#         self.upperarm_ref_log.append(q_ref_dict[upperarm_id])
#         self.upperarm_act_log.append(q_act[upperarm_id])
#         self.wrist_ref_log.append(q_ref_dict[wrist_id])
#         self.wrist_act_log.append(q_act[wrist_id])
#         self.tau_shoulder_log.append(tau[0])
#         self.tau_upperarm_log.append(tau[1])
#         self.tau_wrist_log.append(tau[2])

#     def get_arrays(self):
#         """
#         Get all logged data as numpy arrays
        
#         Returns:
#             dict: All logged data as numpy arrays
#         """
#         return {
#             'time': np.array(self.time_log),
#             'shoulder_ref': np.array(self.shoulder_ref_log),
#             'shoulder_act': np.array(self.shoulder_act_log),
#             'upperarm_ref': np.array(self.upperarm_ref_log),
#             'upperarm_act': np.array(self.upperarm_act_log),
#             'wrist_ref': np.array(self.wrist_ref_log),
#             'wrist_act': np.array(self.wrist_act_log),
#             'tau_shoulder': np.array(self.tau_shoulder_log),
#             'tau_upperarm': np.array(self.tau_upperarm_log),
#             'tau_wrist': np.array(self.tau_wrist_log)
#         }

#     def compute_tracking_errors(self):
#         """
#         Compute tracking errors
        
#         Returns:
#             dict: Tracking error arrays
#         """
#         arrays = self.get_arrays()
        
#         return {
#             'shoulder_error': arrays['shoulder_ref'] - arrays['shoulder_act'],
#             'upperarm_error': arrays['upperarm_ref'] - arrays['upperarm_act'],
#             'wrist_error': arrays['wrist_ref'] - arrays['wrist_act']
#         }

#     def print_statistics(self, joint_names, wave_start_time):
#         """
#         Print tracking performance statistics
        
#         Args:
#             joint_names: List of [shoulder_name, upperarm_name, wrist_name]
#             wave_start_time: Time when waving phase starts [s]
#         """
#         arrays = self.get_arrays()
#         errors = self.compute_tracking_errors()
        
#         print("\n" + "=" * 60)
#         print("📊 Tracking Performance Statistics (MPC)")
#         print("=" * 60)
        
#         # Shoulder statistics
#         shoulder_error = errors['shoulder_error']
#         print(f"\n[Shoulder Joint - {joint_names[0]}]")
#         print(f"RMSE: {np.sqrt(np.mean(shoulder_error**2)):.6f} rad "
#               f"({np.rad2deg(np.sqrt(np.mean(shoulder_error**2))):.3f} deg)")
#         print(f"Max Error: {np.max(np.abs(shoulder_error)):.6f} rad "
#               f"({np.rad2deg(np.max(np.abs(shoulder_error))):.3f} deg)")
#         print(f"Final Error: {shoulder_error[-1]:.6f} rad "
#               f"({np.rad2deg(shoulder_error[-1]):.3f} deg)")
        
#         # Upper arm statistics (waving phase only)
#         wave_start_idx = np.argmin(np.abs(arrays['time'] - wave_start_time))
#         upperarm_error_wave = errors['upperarm_error'][wave_start_idx:]
#         wrist_error_wave = errors['wrist_error'][wave_start_idx:]
        
#         print(f"\n[Upper Arm Joint - {joint_names[1]} - Waving Phase]")
#         print(f"RMSE: {np.sqrt(np.mean(upperarm_error_wave**2)):.6f} rad "
#               f"({np.rad2deg(np.sqrt(np.mean(upperarm_error_wave**2))):.3f} deg)")
#         print(f"Max Error: {np.max(np.abs(upperarm_error_wave)):.6f} rad "
#               f"({np.rad2deg(np.max(np.abs(upperarm_error_wave))):.3f} deg)")
        
#         print(f"\n[Wrist Joint - {joint_names[2]} - Waving Phase]")
#         print(f"RMSE: {np.sqrt(np.mean(wrist_error_wave**2)):.6f} rad "
#               f"({np.rad2deg(np.sqrt(np.mean(wrist_error_wave**2))):.3f} deg)")
#         print(f"Max Error: {np.max(np.abs(wrist_error_wave)):.6f} rad "
#               f"({np.rad2deg(np.max(np.abs(wrist_error_wave))):.3f} deg)")
        
#         # Short summary RMSE in degrees
#         print("\n# Summary RMSE (degrees)")
#         print(f"[Shoulder] RMSE: {np.rad2deg(np.sqrt(np.mean(shoulder_error**2))):.3f} deg")
#         print(f"[Upper Arm] RMSE: {np.rad2deg(np.sqrt(np.mean(upperarm_error_wave**2))):.3f} deg")
#         print(f"[Wrist] RMSE: {np.rad2deg(np.sqrt(np.mean(wrist_error_wave**2))):.3f} deg")
        
#         # Control effort
#         print("\n[Control Effort]")
#         print(f"Mean |Shoulder Torque|: {np.mean(np.abs(arrays['tau_shoulder'])):.3f} Nm")
#         print(f"Mean |Upper Arm Torque|: {np.mean(np.abs(arrays['tau_upperarm'])):.3f} Nm")
#         print(f"Mean |Wrist Torque|: {np.mean(np.abs(arrays['tau_wrist'])):.3f} Nm")
        
#         print("=" * 60)

"""
Data Logger for MPC Simulation

시뮬레이션 중 데이터를 수집하고 저장합니다.
"""

import numpy as np
from .config import PathConfig


class DatasetCollector:
    """
    Residual Torque Dataset Collector

    MPC 시뮬레이션 중 상태, 토크, residual 토크를 수집합니다.
    """

    def __init__(self):
        """Initialize empty data lists"""
        self.dataset_q = []
        self.dataset_qdot = []
        self.dataset_tau = []
        self.dataset_delta_tau = []

    def add_sample(self, q, qdot, tau_mpc, delta_tau):
        """
        Add one sample to dataset

        Args:
            q: Joint positions for controlled joints          [1]  # 수정! [3] → [1]
            qdot: Joint velocities for controlled joints      [1]  # 수정! [3] → [1]
            tau_mpc: MPC torque                               [1]  # 수정! [3] → [1]
            delta_tau: Residual torque (true - MPC)           [1]  # 수정! [3] → [1]
        """
        self.dataset_q.append(q.copy())
        self.dataset_qdot.append(qdot.copy())
        self.dataset_tau.append(tau_mpc.copy())
        self.dataset_delta_tau.append(delta_tau.copy())

    def save_dataset(self, filepath=None):
        """
        Save collected dataset to file

        Args:
            filepath: Path to save .npz file (uses config default if None)

        Returns:
            filepath: Path where data was saved
        """
        if filepath is None:
            filepath = PathConfig.DATASET_PATH

        np.savez(
            filepath,
            q=np.array(self.dataset_q),
            qdot=np.array(self.dataset_qdot),
            tau_mpc=np.array(self.dataset_tau),
            delta_tau=np.array(self.dataset_delta_tau)
        )

        print(f"\n✅ Dataset saved to {filepath}")
        print(f"   Samples: {len(self.dataset_q)}")

        return filepath

    def get_statistics(self):
        """
        Get dataset statistics

        Returns:
            dict: Statistics including mean, max, and ratio of residual torques
        """
        if len(self.dataset_q) == 0:
            return None

        delta_tau = np.array(self.dataset_delta_tau)
        tau_mpc = np.array(self.dataset_tau)

        mean_abs = np.mean(np.abs(delta_tau), axis=0)
        max_abs = np.max(np.abs(delta_tau), axis=0)
        std_abs = np.std(np.abs(delta_tau), axis=0)

        tau_mpc_mean = np.mean(np.abs(tau_mpc), axis=0)
        ratio = mean_abs / (tau_mpc_mean + 1e-6)

        return {
            'mean_abs': mean_abs,
            'max_abs': max_abs,
            'std_abs': std_abs,
            'ratio': ratio,
            'n_samples': len(self.dataset_q)
        }


class TorqueComparisonLogger:
    """
    Torque Comparison Logger — 1DOF (shoulder)

    6가지 토크를 기록합니다:
    1. tau_mpc: MPC가 계산한 토크
    2. tau_true: 실제 필요한 토크 (from residual calculator)
    3. tau_residual: tau_true - tau_mpc
    4. delta_trained: 학습된 NN이 예측한 residual
    5. tau_total: tau_mpc + delta_trained (실제 적용된 토크)
    6. tau_final: tau_mpc + alpha * clamp(delta_trained)
    """

    def __init__(self):
        """Initialize empty log lists"""
        self.time_log = []

        # 수정! 주석: (n_timesteps, 3) → (n_timesteps, 1)
        self.tau_mpc_log = []          # 1. MPC 토크
        self.tau_true_log = []         # 2. 실제 필요 토크
        self.tau_residual_log = []     # 3. Residual (true - mpc)
        self.delta_trained_log = []    # 4. 학습된 NN 예측
        self.tau_total_log = []        # 5. MPC + delta_trained
        self.tau_final_log = []        # 6. MPC + alpha*clamp(delta)

    def add_sample(self, t, tau_mpc, tau_true, delta_trained, tau_final):
        """
        Add one sample to log

        Args:
            t: Time [s]
            tau_mpc: MPC torque                [1]  # 수정! [3] → [1]
            tau_true: True required torque     [1]  # 수정! [3] → [1]
            delta_trained: NN predicted residual [1]  # 수정! [3] → [1]
            tau_final: Final applied torque    [1]  # 수정! [3] → [1]
        """
        self.time_log.append(t)
        self.tau_mpc_log.append(tau_mpc.copy())
        self.tau_true_log.append(tau_true.copy())
        self.tau_residual_log.append((tau_true - tau_mpc).copy())
        self.delta_trained_log.append(delta_trained.copy())
        self.tau_total_log.append((tau_mpc + delta_trained).copy())
        self.tau_final_log.append(tau_final.copy())

    def get_arrays(self):
        """
        Get all logged data as numpy arrays

        Returns:
            dict: All logged data as numpy arrays
        """
        return {
            'time':           np.array(self.time_log),
            'tau_mpc':        np.array(self.tau_mpc_log),           # (N, 1)  # 수정!
            'tau_true':       np.array(self.tau_true_log),          # (N, 1)  # 수정!
            'tau_residual':   np.array(self.tau_residual_log),      # (N, 1)  # 수정!
            'delta_trained':  np.array(self.delta_trained_log),     # (N, 1)  # 수정!
            'tau_total':      np.array(self.tau_total_log),         # (N, 1)  # 수정!
            'tau_final':      np.array(self.tau_final_log)          # (N, 1)  # 수정!
        }


class TrackingLogger:
    """
    Tracking Performance Logger — 1DOF (shoulder)

    제어 성능 추적을 위한 데이터를 수집합니다.
    """

    def __init__(self):
        """Initialize empty log lists"""
        self.time_log = []
        # 수정! upperarm_*, wrist_* 로그 전체 제거
        self.shoulder_ref_log = []
        self.shoulder_act_log = []
        self.tau_shoulder_log = []

    def add_sample(self, t, q_ref_dict, q_act, tau, shoulder_id):  # 수정! upperarm_id, wrist_id 제거
        """
        Add one sample to log

        Args:
            t: Time [s]
            q_ref_dict: Reference angles dictionary {shoulder_id: angle}
            q_act: Actual joint positions (full qpos)
            tau: Applied torques for controlled joints  [1]
            shoulder_id: qpos index for shoulder
        """
        # 수정! shoulder만 기록
        self.time_log.append(t)
        self.shoulder_ref_log.append(q_ref_dict[shoulder_id])
        self.shoulder_act_log.append(q_act[shoulder_id])
        self.tau_shoulder_log.append(tau[0])

    def get_arrays(self):
        """
        Get all logged data as numpy arrays

        Returns:
            dict: All logged data as numpy arrays
        """
        # 수정! upperarm, wrist 키 제거
        return {
            'time':          np.array(self.time_log),
            'shoulder_ref':  np.array(self.shoulder_ref_log),
            'shoulder_act':  np.array(self.shoulder_act_log),
            'tau_shoulder':  np.array(self.tau_shoulder_log)
        }

    def compute_tracking_errors(self):
        """
        Compute tracking errors

        Returns:
            dict: Tracking error arrays
        """
        arrays = self.get_arrays()

        # 수정! shoulder_error만 반환
        return {
            'shoulder_error': arrays['shoulder_ref'] - arrays['shoulder_act']
        }

    def print_statistics(self, shoulder_name,start_time=0.0):  # 수정! joint_names리스트, wave_start_time 제거 → shoulder_name만
        """
        Print tracking performance statistics

        Args:
            shoulder_name: Joint name string for display
        """
        # valid_indices = [i for i, t in enumerate(self.time) if t >= start_time]

        arrays = self.get_arrays()
        errors = self.compute_tracking_errors()

        print("\n" + "=" * 60)
        print("📊 Tracking Performance Statistics (MPC) — Shoulder Only")  # 수정!
        print("=" * 60)

        # 수정! Shoulder 통계만
        shoulder_error = errors['shoulder_error']

        print(f"\n[Shoulder Joint - {shoulder_name}]")
        print(f"  RMSE:        {np.sqrt(np.mean(shoulder_error**2)):.6f} rad "
              f"({np.rad2deg(np.sqrt(np.mean(shoulder_error**2))):.3f} deg)")
        print(f"  Max Error:   {np.max(np.abs(shoulder_error)):.6f} rad "
              f"({np.rad2deg(np.max(np.abs(shoulder_error))):.3f} deg)")
        print(f"  Final Error: {shoulder_error[-1]:.6f} rad "
              f"({np.rad2deg(shoulder_error[-1]):.3f} deg)")

        # 수정! upperarm/wrist 통계 블록 전체 제거

        # Summary
        print(f"\n# Summary RMSE (degrees)")
        print(f"  [Shoulder] RMSE: {np.rad2deg(np.sqrt(np.mean(shoulder_error**2))):.3f} deg")

        # Control effort
        print(f"\n[Control Effort]")
        print(f"  Mean |Shoulder Torque|: {np.mean(np.abs(arrays['tau_shoulder'])):.3f} Nm")

        print("=" * 60)
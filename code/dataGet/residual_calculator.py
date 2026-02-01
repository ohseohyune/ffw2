"""
Residual Torque Calculator

실제 필요한 토크와 MPC 토크의 차이(residual)를 계산합니다.
"""

import numpy as np
import mujoco


class ResidualCalculator:
    """
    Calculate residual torque (true required torque - MPC torque)
    
    실제로 필요한 토크와 MPC가 출력한 토크의 차이를 계산합니다.
    """
    
    def __init__(self, model, controlled_joint_ids):
        """
        Args:
            model: MuJoCo model
            controlled_joint_ids: List of qpos indices for controlled joints
        """
        self.model = model
        self.controlled_joint_ids = controlled_joint_ids
        self.nq = len(controlled_joint_ids)
        
        # Create dedicated data for residual calculation
        self.data_temp = mujoco.MjData(model)

    def compute_residual(self, data, q_k, qdot_k, tau_mpc, dt):
        # 1. 실제 속도 변화를 통한 가속도 계산 (수치 미분)
        qdot_k1 = data.qvel.copy()
        # q_k, qdot_k는 step 이전의 데이터, qdot_k1은 step 이후의 데이터입니다.
        qddot_real_full = (qdot_k1 - qdot_k) / dt
        qddot_real = qddot_real_full[self.controlled_joint_ids]

        # 2. dynamics 계산을 위한 이전 상태(k) 설정
        self.data_temp.qpos[:] = q_k
        self.data_temp.qvel[:] = qdot_k
        
        # mj_forward를 통해 Mass Matrix(M)와 Bias Force(C+G) 계산
        mujoco.mj_forward(self.model, self.data_temp)
        
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data_temp.qM)
        M_sub = M_full[np.ix_(self.controlled_joint_ids, self.controlled_joint_ids)]
        
        bias_full = self.data_temp.qfrc_bias.copy()
        bias_sub = bias_full[self.controlled_joint_ids]

        # 3. 실제 필요했던 토크 계산: τ_true = M(k)@qddot_real + bias(k)
        tau_true = M_sub @ qddot_real + bias_sub
        
        # 4. Residual 계산
        tau_residual = tau_mpc - tau_true

        return tau_residual

    def compute_residual_batch(self, trajectories, dt):
        """
        Compute residuals for a batch of trajectories
        
        Args:
            trajectories: List of trajectory dicts with keys:
                         'q_full', 'qdot_full', 'tau_mpc'
            dt: Time step
        
        Returns:
            residuals: Array of residual torques [N, 3]
        """
        residuals = []
        
        # Create temporary data for dynamics computation
        data_temp = mujoco.MjData(self.model)
        
        for traj in trajectories:
            q_full = traj['q_full']
            qdot_full = traj['qdot_full']
            tau_mpc = traj['tau_mpc']
            
            # Set state
            data_temp.qpos[:] = q_full
            data_temp.qvel[:] = qdot_full
            
            # Compute dynamics
            mujoco.mj_forward(self.model, data_temp)
            
            # Get mass matrix and bias
            M_full = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, M_full, data_temp.qM)
            M_sub = M_full[np.ix_(self.controlled_joint_ids, self.controlled_joint_ids)]
            
            bias_full = data_temp.qfrc_bias.copy()
            bias_sub = bias_full[self.controlled_joint_ids]
            
            # Solve for required acceleration
            qddot_req = np.linalg.solve(M_sub, tau_mpc - bias_sub)
            print("DEBUG compute_residual_batch qddot_req:", qddot_req)
            
            # Calculate required torque
            tau_real = M_sub @ qddot_req + bias_sub
            print("DEBUG compute_residual_batch tau_real:", tau_real)
            
            # Residual
            residual = tau_real - tau_mpc
            residuals.append(residual)
        
        return np.array(residuals)


def create_residual_calculator(model, controlled_joint_ids):
    """
    Factory function to create ResidualCalculator
    
    Args:
        model: MuJoCo model
        controlled_joint_ids: List of qpos indices for controlled joints
    
    Returns:
        ResidualCalculator instance
    """
    return ResidualCalculator(model, controlled_joint_ids)
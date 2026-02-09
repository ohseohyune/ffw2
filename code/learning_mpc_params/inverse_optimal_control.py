# 핵심 IOC 알고리즘

"""
Inverse Optimal Control using Relaxed KKT Conditions

전문가 시연 데이터로부터 MPC 비용함수 파라미터를 역으로 추정합니다.
KKT 조건 완화(Relaxed KKT) 방법을 사용합니다.

핵심 아이디어:
    최적성 조건 ∇L = 0을 정확히 만족시키는 대신,
    ‖∇L‖²을 최소화하는 파라미터 θ를 찾습니다.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List


class InverseOptimalControl:
    """
    KKT 조건 완화를 이용한 역최적제어
    
    전문가 시연으로부터 MPC 비용함수 가중치를 학습합니다.
    """
    
    def __init__(self, model, joint_ids, horizon, dt):
        """
        Args:
            model: MuJoCo model
            joint_ids: 제어할 관절 인덱스
            horizon: MPC 예측 구간
            dt: 시간 간격
        """
        self.model = model
        self.joint_ids = joint_ids
        self.nq = len(joint_ids)
        self.horizon = horizon
        self.dt = dt
        
        # MPC 동역학 계산용 데이터
        self.data_temp = mujoco.MjData(model)
        
        # 파라미터 초기값 저장
        self.theta_init = None
        
    def load_demonstration_data(self, data_path):
        """
        시연 데이터 로드
        
        Args:
            data_path: .npz 파일 경로
            
        Returns:
            demonstrations: List of dicts with keys:
                - 't': time
                - 'q': joint positions
                - 'qdot': joint velocities  
                - 'u': control inputs (tau)
                - 'q_ref': reference positions
        """
        data = np.load(data_path)
        
        # 시연 데이터를 시간 구간으로 분할
        demonstrations = []
        
        # 전체 데이터를 horizon 길이의 세그먼트로 나눔
        n_samples = len(data['q'])
        segment_length = self.horizon + 1
        
        for start_idx in range(0, n_samples - segment_length, segment_length // 2):
            end_idx = start_idx + segment_length
            
            demo = {
                'q': data['q'][start_idx:end_idx],
                'qdot': data['qdot'][start_idx:end_idx],
                'u': data['tau_mpc'][start_idx:end_idx],
                't': np.arange(segment_length) * self.dt
            }
            
            demonstrations.append(demo)
        
        print(f"✅ Loaded {len(demonstrations)} demonstration segments")
        return demonstrations
    
    def compute_gradient_norm(self, theta, demonstration):
        """
        주어진 파라미터 θ에 대해 ‖∇L‖² 계산
        
        이것이 목적함수입니다. 이 값을 최소화하는 θ를 찾습니다.

        주어진 비용 가중치 θ가 있을 때, 전문가 시연이 그 비용에 대해 ‘거의 최적’이었는지를 KKT 조건의 잔차(∇L) 크기로 평가한다.
        
        Args:
            theta: 비용함수 파라미터 [q_pos, q_vel, r_tau, q_terminal] (4개)
            demonstration: 시연 데이터 dict
            
        Returns:
            gradient_norm_squared: ‖∇L‖² 값
        """
        # 파라미터 언팩
        q_pos_weight = theta[0]
        q_vel_weight = theta[1] 
        r_tau_weight = theta[2]
        q_terminal_weight = theta[3]
        
        # 시연 데이터 추출
        q_demo = demonstration['q']      # (horizon+1, nq)
        qdot_demo = demonstration['qdot']
        u_demo = demonstration['u']      # (horizon+1, nq)
        
        # Lagrangian 기울기를 저장할 배열
        grad_L = np.zeros((self.horizon, self.nq))
        
        # 각 시간 스텝에서 Lagrangian의 u에 대한 편미분 계산
        for k in range(self.horizon):
            # 현재 상태
            q_k = q_demo[k]
            qdot_k = qdot_demo[k]
            u_k = u_demo[k]
            
            # 다음 상태 (실제 시연)
            q_next_demo = q_demo[k+1]
            qdot_next_demo = qdot_demo[k+1]
            
            # ∂l/∂u 계산 (stage cost의 입력에 대한 미분) -> 어떤 입력이 가장 cost를 작게 하는지 계산 
            # l = q_pos * ‖q - q_ref‖² + q_vel * ‖qdot‖² + r_tau * ‖u‖²
            dldu_direct = 2 * r_tau_weight * u_k
            
            # ∂(next state)/∂u 계산 (동역학 제약의 u에 대한 미분)
            # x_{k+1} = f(x_k, u_k)이므로 ∂x_{k+1}/∂u_k를 구해야 함
            
            # 수치 미분으로 근사
            epsilon = 1e-6
            du = np.eye(self.nq) * epsilon
            
            # 동역학 계산을 위한 상태 설정
            dynamics_grad = np.zeros((self.nq, self.nq))
            
            for i in range(self.nq):
                # u + epsilon
                u_plus = u_k.copy()
                u_plus[i] += epsilon
                q_next_plus, qdot_next_plus = self._forward_dynamics(q_k, qdot_k, u_plus)
                
                # u - epsilon  
                u_minus = u_k.copy()
                u_minus[i] -= epsilon
                q_next_minus, qdot_next_minus = self._forward_dynamics(q_k, qdot_k, u_minus)
                
                # 수치 미분
                dq_next_du = (q_next_plus - q_next_minus) / (2 * epsilon)
                dqdot_next_du = (qdot_next_plus - qdot_next_minus) / (2 * epsilon)
                
                dynamics_grad[:, i] = dq_next_du
            
            # 다음 스텝의 비용에 대한 영향 (체인 룰)
            # ∂l_{k+1}/∂q_{k+1} * ∂q_{k+1}/∂u_k
            if k < self.horizon - 1:
                # 참조값 (여기서는 시연의 마지막 상태를 목표로 가정)
                q_ref = q_demo[-1]
                q_error_next = q_demo[k+1] - q_ref
                
                dldu_chain = dynamics_grad.T @ (2 * q_pos_weight * q_error_next)
                dldu_chain += dynamics_grad.T @ (2 * q_vel_weight * qdot_demo[k+1])
            else:
                # 종단 비용
                q_ref = q_demo[-1]
                q_error_terminal = q_demo[-1] - q_ref
                dldu_chain = dynamics_grad.T @ (2 * q_terminal_weight * q_error_terminal)
            
            # 전체 기울기
            grad_L[k] = dldu_direct + dldu_chain
        
        # ‖∇L‖² 계산
        gradient_norm_squared = np.sum(grad_L ** 2)
        
        return gradient_norm_squared
    
    def _forward_dynamics(self, q, qdot, u):
        """
        한 스텝 동역학 시뮬레이션 (RK4)
        
        Args:
            q: 관절 위치
            qdot: 관절 속도
            u: 제어 입력
            
        Returns:
            q_next: 다음 위치
            qdot_next: 다음 속도
        """
        # MuJoCo 상태 설정
        self.data_temp.qpos[self.joint_ids] = q
        self.data_temp.qvel[self.joint_ids] = qdot
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data_temp)
        
        # Mass matrix
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data_temp.qM)
        M = M_full[np.ix_(self.joint_ids, self.joint_ids)]
        
        # Bias force
        bias = self.data_temp.qfrc_bias[self.joint_ids]
        
        # 가속도 계산
        qddot = np.linalg.solve(M, u - bias)
        
        # RK4 적분
        k1_v = qdot
        k1_a = qddot
        
        k2_v = qdot + 0.5 * self.dt * k1_a
        k2_a = qddot  # 간단화: 같은 가속도 사용
        
        k3_v = qdot + 0.5 * self.dt * k2_a
        k3_a = qddot
        
        k4_v = qdot + self.dt * k3_a
        k4_a = qddot
        
        q_next = q + (self.dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        qdot_next = qdot + (self.dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + 4*k4_a)
        
        return q_next, qdot_next
    
    def learn_cost_weights(self, demonstrations, theta_init=None):
        """
        시연 데이터로부터 비용함수 가중치 학습
        
        Args:
            demonstrations: 시연 데이터 리스트
            theta_init: 초기 파라미터 (None이면 자동 설정)
            
        Returns:
            theta_opt: 최적 파라미터
            result: 최적화 결과
        """
        # 초기 파라미터 설정
        if theta_init is None:
            # 합리적인 초기값
            theta_init = np.array([
                1000.0,  # q_pos_weight
                50.0,    # q_vel_weight  
                0.01,    # r_tau_weight
                1500.0   # q_terminal_weight
            ])
        
        self.theta_init = theta_init.copy()
        
        print("\n" + "="*60)
        print("🎯 Inverse Optimal Control: Learning MPC Cost Weights")
        print("="*60)
        print(f"Method: Relaxed KKT (minimize ‖∇L‖²)")
        print(f"Demonstrations: {len(demonstrations)} segments")
        print(f"Horizon: {self.horizon}")
        print(f"Initial parameters: {theta_init}")
        
        # 목적함수: 모든 시연에 대한 평균 ‖∇L‖²
        def objective(theta):
            total_grad_norm = 0.0
            for demo in demonstrations:
                grad_norm_sq = self.compute_gradient_norm(theta, demo)
                total_grad_norm += grad_norm_sq
            
            avg_grad_norm = total_grad_norm / len(demonstrations)
            return avg_grad_norm
        
        # 제약조건: 모든 가중치는 양수
        bounds = [(1e-3, None)] * len(theta_init)
        
        # 정규화 제약: 합이 일정 (스케일 고정)
        # 이것이 없으면 모든 가중치가 0으로 수렴할 수 있음
        total_weight = np.sum(theta_init)
        constraints = {
            'type': 'eq',
            'fun': lambda theta: np.sum(theta) - total_weight
        }
        
        print("\n▶️  Starting optimization...")
        print(f"   Bounds: all weights > 0")
        print(f"   Constraint: Σθ = {total_weight:.1f}")
        
        # 최적화 실행
        result = minimize(
            objective,
            theta_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 100,
                'ftol': 1e-6,
                'disp': True
            }
        )
        
        theta_opt = result.x
        
        print("\n✅ Optimization completed!")
        print(f"   Success: {result.success}")
        print(f"   Final ‖∇L‖²: {result.fun:.6e}")
        print(f"   Iterations: {result.nit}")
        
        return theta_opt, result
    
    def compare_parameters(self, theta_learned, theta_original=None):
        """
        학습된 파라미터와 원래 파라미터 비교
        
        Args:
            theta_learned: 학습된 파라미터
            theta_original: 원래 파라미터 (None이면 초기값 사용)
        """
        if theta_original is None:
            theta_original = self.theta_init
        
        param_names = ['Q_pos', 'Q_vel', 'R_tau', 'Q_terminal']
        
        print("\n" + "="*60)
        print("📊 Parameter Comparison")
        print("="*60)
        print(f"{'Parameter':<15} {'Original':>12} {'Learned':>12} {'Ratio':>10}")
        print("-"*60)
        
        for i, name in enumerate(param_names):
            orig = theta_original[i]
            learned = theta_learned[i]
            ratio = learned / orig if orig != 0 else float('inf')
            
            print(f"{name:<15} {orig:>12.2f} {learned:>12.2f} {ratio:>10.2f}x")
        
        print("="*60)
    
    def visualize_results(self, theta_learned, demonstrations, n_samples=3):
        """
        학습 결과 시각화
        
        Args:
            theta_learned: 학습된 파라미터
            demonstrations: 시연 데이터
            n_samples: 시각화할 시연 개수
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 샘플 선택
        demo_indices = np.linspace(0, len(demonstrations)-1, n_samples, dtype=int)
        
        for demo_idx in demo_indices:
            demo = demonstrations[demo_idx]
            
            # 기울기 계산
            grad_norm_init = self.compute_gradient_norm(self.theta_init, demo)
            grad_norm_learned = self.compute_gradient_norm(theta_learned, demo)
            
            # 시연 데이터
            t = demo['t']
            q = demo['q'][:, 0]  # 첫 번째 관절만
            qdot = demo['qdot'][:, 0]
            u = demo['u'][:, 0]
            
            # Plot 1: Position trajectory
            axes[0, 0].plot(t, q, alpha=0.6, label=f'Demo {demo_idx}')
            
            # Plot 2: Velocity  
            axes[0, 1].plot(t, qdot, alpha=0.6)
            
            # Plot 3: Control input
            axes[1, 0].plot(t, u, alpha=0.6)
        
        # Plot 4: Gradient norm comparison
        grad_norms_init = []
        grad_norms_learned = []
        
        for demo in demonstrations:
            grad_norms_init.append(
                np.sqrt(self.compute_gradient_norm(self.theta_init, demo))
            )
            grad_norms_learned.append(
                np.sqrt(self.compute_gradient_norm(theta_learned, demo))
            )
        
        x = np.arange(len(demonstrations))
        axes[1, 1].bar(x - 0.2, grad_norms_init, 0.4, 
                      label='Initial θ', alpha=0.7)
        axes[1, 1].bar(x + 0.2, grad_norms_learned, 0.4,
                      label='Learned θ', alpha=0.7)
        
        # 레이블 설정
        axes[0, 0].set_ylabel('Position [rad]')
        axes[0, 0].set_title('Demonstration Trajectories - Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_ylabel('Velocity [rad/s]')
        axes[0, 1].set_title('Demonstration Trajectories - Velocity')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Torque [Nm]')
        axes[1, 0].set_title('Demonstration Trajectories - Control')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Demonstration Index')
        axes[1, 1].set_ylabel('‖∇L‖')
        axes[1, 1].set_title('KKT Gradient Norm Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/ioc_results.png', dpi=150)
        print("\n✅ Visualization saved: ioc_results.png")
        plt.show()


def apply_learned_weights_to_mpc(mpc_controller, theta_learned):
    """
    학습된 가중치를 MPC 컨트롤러에 적용
    
    Args:
        mpc_controller: TorqueMPC 인스턴스
        theta_learned: 학습된 파라미터 [q_pos, q_vel, r_tau, q_terminal]
    """
    nq = mpc_controller.nq
    
    Q_pos = np.eye(nq) * theta_learned[0]
    Q_vel = np.eye(nq) * theta_learned[1]
    R_tau = np.eye(nq) * theta_learned[2]
    Q_terminal = np.eye(nq) * theta_learned[3]
    
    mpc_controller.update_cost_weights(
        Q_pos=Q_pos,
        Q_vel_ref=Q_vel,
        R_tau=R_tau,
        Q_terminal=Q_terminal
    )
    
    print("\n✅ Learned weights applied to MPC controller!")
    print(f"   Q_pos: {theta_learned[0]:.2f}")
    print(f"   Q_vel: {theta_learned[1]:.2f}")
    print(f"   R_tau: {theta_learned[2]:.2f}")
    print(f"   Q_terminal: {theta_learned[3]:.2f}")
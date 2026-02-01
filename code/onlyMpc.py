import numpy as np
import mujoco
import mujoco.viewer
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ===============================
# Torque-based MPC Controller
# ===============================
class TorqueMPC:
    """
    Model Predictive Control for torque-based robot control
    
    State: x = [q, qdot] (joint positions and velocities)
    Input: u = tau (joint torques)
    
    Dynamics: M(q) * qddot + C(q, qdot) + G(q) = tau
              -> qddot = M^-1 * (tau - bias)
              where bias = C(q, qdot) + G(q)
    
    Cost function:
        J = Σ[k=0 to H-1] { (q_k - q_ref)^T Q (q_k - q_ref) + tau_k^T R tau_k }
            + (q_H - q_ref)^T Q_terminal (q_H - q_ref)
    
    where:
        - Q: position tracking weight (상태 추종 가중치)
        - R: control effort weight (제어 입력 가중치, 에너지 최소화)
        - Q_terminal: terminal state weight (최종 상태 가중치, 안정성)
    """
    
    def __init__(self, model, joint_ids, horizon=10, dt=0.002):
        """
        Args:
            model: MuJoCo model
            joint_ids: List of joint indices to control (e.g., [10, 12, 18])
            horizon: MPC prediction horizon
            dt: Time step for prediction
        """
        self.model = model
        self.joint_ids = joint_ids
        self.nq = len(joint_ids)  # Number of controlled joints
        self.horizon = horizon
        self.dt = dt
        
        # Cost weights (튜닝 가능한 핵심 파라미터)
        self.Q_pos = np.eye(self.nq) * 500.0       # Position tracking weight
        self.R_tau = np.eye(self.nq) * 0.01        # Torque penalty (energy minimization)
        self.Q_terminal = np.eye(self.nq) * 800.0  # Terminal cost (stability)
        
        # Torque limits (물리적 제약)
        self.tau_max = 100.0 * np.ones(self.nq)
        self.tau_min = -self.tau_max
        
        # Dedicated data for MPC computation (thread-safe)
        self.data_mpc = mujoco.MjData(model)
        
        # Cached dynamics
        self.M_inv_cache = None
        self.bias_cache = None
        
        # Full DOF mapping (전체 DOF에서 제어 관절만 추출)
        self.full_nv = model.nv
        
    def _cache_dynamics_from_state(self, q_full, qdot_full):
        """
        Cache dynamics (M^-1, bias) from full state
        
        MuJoCo의 dynamics를 계산하여 캐싱:
        - M(q): Mass matrix (inertia)
        - bias = qfrc_bias: Coriolis, centrifugal, gravity forces
        """
        # Set full state
        self.data_mpc.qpos[:] = q_full
        self.data_mpc.qvel[:] = qdot_full
        
        # Compute consistent dynamics
        mujoco.mj_forward(self.model, self.data_mpc)
        
        # Extract full mass matrix
        M_full = np.zeros((self.full_nv, self.full_nv))
        mujoco.mj_fullM(self.model, M_full, self.data_mpc.qM)
        
        # Extract submatrix for controlled joints
        M_sub = M_full[np.ix_(self.joint_ids, self.joint_ids)]
        bias_full = self.data_mpc.qfrc_bias.copy()
        bias_sub = bias_full[self.joint_ids]
        
        # Cache
        self.M_inv_cache = np.linalg.inv(M_sub)
        self.bias_cache = bias_sub
        
    def _predict_state(self, q, qdot, tau):
        """
        Forward Euler integration for one timestep
        
        Dynamics:
            qddot = M^-1 * (tau - bias)
            q_next = q + dt * qdot
            qdot_next = qdot + dt * qddot
        
        Returns:
            (q_next, qdot_next)
        """
        qddot = self.M_inv_cache @ (tau - self.bias_cache)
        q_next = q + self.dt * qdot
        qdot_next = qdot + self.dt * qddot
        return q_next, qdot_next
    
    def _compute_cost(self, tau_seq, q0, qdot0, q_ref):
        """
        Compute total cost for given torque sequence
        
        Cost = Σ stage_cost + terminal_cost
        
        where:
            stage_cost = (q - q_ref)^T Q (q - q_ref) + tau^T R tau
            terminal_cost = (q_H - q_ref)^T Q_terminal (q_H - q_ref)
        """
        q, qdot = q0.copy(), qdot0.copy()
        cost = 0.0
        tau_seq = tau_seq.reshape(self.horizon, self.nq)
        
        # Stage costs
        for k in range(self.horizon):
            # Position tracking cost
            q_error = q - q_ref
            cost += q_error.T @ self.Q_pos @ q_error
            
            # Control effort cost (에너지 최소화)
            cost += tau_seq[k].T @ self.R_tau @ tau_seq[k]
            
            # Predict next state
            q, qdot = self._predict_state(q, qdot, tau_seq[k])
        
        # Terminal cost (최종 상태 안정성)
        q_error_final = q - q_ref
        cost += q_error_final.T @ self.Q_terminal @ q_error_final
        
        return cost
    
    def compute_control(self, q_full, qdot_full, q_ref_sub):
        """
        Solve MPC optimization problem
        
        Args:
            q_full: Full joint positions (전체 DOF)
            qdot_full: Full joint velocities
            q_ref_sub: Reference positions for controlled joints only
        
        Returns:
            tau_total: Total torque including gravity compensation
        """
        # Extract controlled joints
        q0 = q_full[self.joint_ids]
        qdot0 = qdot_full[self.joint_ids]
        
        # Cache dynamics
        self._cache_dynamics_from_state(q_full, qdot_full)
        
        # Initial guess: zero torque
        tau_init = np.zeros(self.horizon * self.nq)
        
        # Bounds: torque limits
        bounds = [(self.tau_min[i % self.nq], self.tau_max[i % self.nq])
                  for i in range(self.horizon * self.nq)]
        
        # Solve optimization
        result = minimize(
            lambda tau: self._compute_cost(tau, q0, qdot0, q_ref_sub),
            tau_init,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-5}
        )
        
        # Extract first control (Receding Horizon)
        tau_opt = result.x[:self.nq]
        
        # Add gravity compensation
        tau_total = tau_opt + self.bias_cache
        
        # Clip to limits
        tau_total = np.clip(tau_total, self.tau_min, self.tau_max)
        
        return tau_total


# ===============================
# Path 설정
# ===============================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")
XML_PATH = os.path.join(MODEL_DIR, "scene_ffw_sg2.xml")

# ===============================
# MuJoCo 모델 로드
# ===============================
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# ===============================
# 제어할 joints 정의 (CRITICAL: actuator IDs 사용!)
# ===============================
# Joint names
shoulder_joint_name = "arm_r_joint1"
upperarm_joint_name = "arm_r_joint3"
wrist_joint_name = "arm_r_joint7"

# Get joint IDs (for reading qpos/qvel)
shoulder_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, shoulder_joint_name)
upperarm_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, upperarm_joint_name)
wrist_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wrist_joint_name)

# Get qpos indices (for MPC state extraction)
shoulder_qpos_id = model.jnt_qposadr[shoulder_joint_id]
upperarm_qpos_id = model.jnt_qposadr[upperarm_joint_id]
wrist_qpos_id = model.jnt_qposadr[wrist_joint_id]

# Controlled joint IDs for MPC (qpos indices)
controlled_joint_ids = [shoulder_qpos_id, upperarm_qpos_id, wrist_qpos_id]

# ⭐ CRITICAL: Get ACTUATOR IDs (for data.ctrl)
motor_names = ["motor_arm_r_joint1", "motor_arm_r_joint3", "motor_arm_r_joint7"]
motor_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
             for name in motor_names]

print("=" * 60)
print("🤖 MPC-based Waving Motion Control")
print("=" * 60)
print(f"Controlled joints: {shoulder_joint_name}, {upperarm_joint_name}, {wrist_joint_name}")
print(f"Joint qpos IDs: {controlled_joint_ids}")
print(f"Motor actuator IDs: {motor_ids}")
print("=" * 60)

# ===============================
# MPC Controller 초기화
# ===============================
controller = TorqueMPC(
    model=model,
    joint_ids=controlled_joint_ids,
    horizon=10,  # 10-step lookahead
    dt=0.002     # 500 Hz
)

# ===============================
# Reference Trajectory 파라미터 (절대 변경 금지)
# ===============================
# 어깨 관절 설정
shoulder_start = 0.0
shoulder_target = -2.8
T_raise = 2.0

# 손목 흔들기 설정
wrist_center = 0.0
wrist_amplitude = 0.4
wave_frequency = 1.0

# 상완 비틀기 설정
upperarm_center = 0.0
upperarm_amplitude = 0.7
phase_delay = 0.0

T_wave = 2.0
T_wait = 0.5

# ===============================
# Logging 변수
# ===============================
time_log = []
shoulder_ref_log = []
shoulder_act_log = []
upperarm_ref_log = []
upperarm_act_log = []
wrist_ref_log = []
wrist_act_log = []
tau_shoulder_log = []
tau_upperarm_log = []
tau_wrist_log = []

# ===============================
# 초기화
# ===============================
mujoco.mj_forward(model, data)
data.ctrl[:] = 0.0  # Torque mode: all zeros initially

# ===============================
# 시뮬레이션 실행
# ===============================
sim_dt = 0.002  # 500 Hz
sim_duration = T_raise + T_wait + T_wave + 1.0
n_steps = int(sim_duration / sim_dt)

print("\n시뮬레이션 시작...")
print(f"Duration: {sim_duration:.1f}s, Steps: {n_steps}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(n_steps):
        t = step * sim_dt
        
        # ===========================
        # Reference Trajectory 생성 (절대 변경 금지)
        # ===========================
        if t <= T_raise:
            # Phase 1: 팔 올리기
            s = np.clip(t / T_raise, 0.0, 1.0)
            shoulder_des = (1 - s) * shoulder_start + s * shoulder_target
            upperarm_des = upperarm_center
            wrist_des = wrist_center
            
        elif t <= T_raise + T_wait:
            # Phase 2: 대기
            shoulder_des = shoulder_target
            upperarm_des = upperarm_center
            wrist_des = wrist_center
            
        elif t <= T_raise + T_wait + T_wave:
            # Phase 3: 손 흔들기
            shoulder_des = shoulder_target
            
            t_wave = t - (T_raise + T_wait)
            wrist_des = wrist_center + wrist_amplitude * np.sin(2 * np.pi * wave_frequency * t_wave)
            
            t_wave_delayed = t_wave - phase_delay
            if t_wave_delayed > 0:
                upperarm_des = upperarm_center + upperarm_amplitude * np.sin(2 * np.pi * wave_frequency * t_wave_delayed)
            else:
                upperarm_des = upperarm_center
        else:
            # Phase 4: 종료 대기
            shoulder_des = shoulder_target
            upperarm_des = upperarm_center
            wrist_des = wrist_center
        
        # Reference for controlled joints
        q_ref = np.array([shoulder_des, upperarm_des, wrist_des])
        
        # ===========================
        # MPC Torque Computation
        # ===========================
        tau_mpc = controller.compute_control(
            q_full=data.qpos.copy(),
            qdot_full=data.qvel.copy(),
            q_ref_sub=q_ref
        )
        
        # ===========================
        # Apply torque using ACTUATOR IDs (FIXED!)
        # ===========================
        data.ctrl[:] = 0.0  # Reset all
        for i, motor_id in enumerate(motor_ids):
            data.ctrl[motor_id] = tau_mpc[i]
        
        # ===========================
        # Simulation Step
        # ===========================
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # ===========================
        # Logging
        # ===========================
        time_log.append(t)
        shoulder_ref_log.append(shoulder_des)
        shoulder_act_log.append(data.qpos[shoulder_qpos_id])
        upperarm_ref_log.append(upperarm_des)
        upperarm_act_log.append(data.qpos[upperarm_qpos_id])
        wrist_ref_log.append(wrist_des)
        wrist_act_log.append(data.qpos[wrist_qpos_id])
        tau_shoulder_log.append(tau_mpc[0])
        tau_upperarm_log.append(tau_mpc[1])
        tau_wrist_log.append(tau_mpc[2])

print("시뮬레이션 완료!\n")

# ===============================
# Plot 1: Joint Tracking
# ===============================
time_log = np.array(time_log)
shoulder_ref_log = np.array(shoulder_ref_log)
shoulder_act_log = np.array(shoulder_act_log)
upperarm_ref_log = np.array(upperarm_ref_log)
upperarm_act_log = np.array(upperarm_act_log)
wrist_ref_log = np.array(wrist_ref_log)
wrist_act_log = np.array(wrist_act_log)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Shoulder
axes[0].plot(time_log, shoulder_ref_log, 'r--', linewidth=2, label="Reference")
axes[0].plot(time_log, shoulder_act_log, 'b-', linewidth=1.5, label="Actual (MPC)")
axes[0].axvline(T_raise, color='gray', linestyle=':', alpha=0.5)
axes[0].axvline(T_raise + T_wait, color='gray', linestyle=':', alpha=0.5)
axes[0].axvline(T_raise + T_wait + T_wave, color='gray', linestyle=':', alpha=0.5)
axes[0].set_ylabel("Shoulder Angle [rad]")
axes[0].set_title("Shoulder Joint (arm_r_joint1) - MPC Control")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Upper Arm
axes[1].plot(time_log, upperarm_ref_log, 'r--', linewidth=2, label="Reference")
axes[1].plot(time_log, upperarm_act_log, 'm-', linewidth=1.5, label="Actual (MPC)")
axes[1].axvline(T_raise, color='gray', linestyle=':', alpha=0.5)
axes[1].axvline(T_raise + T_wait, color='gray', linestyle=':', alpha=0.5)
axes[1].axvline(T_raise + T_wait + T_wave, color='gray', linestyle=':', alpha=0.5)
axes[1].set_ylabel("Upper Arm Angle [rad]")
axes[1].set_title("Upper Arm Joint (arm_r_joint3) - MPC Control")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Wrist
axes[2].plot(time_log, wrist_ref_log, 'r--', linewidth=2, label="Reference")
axes[2].plot(time_log, wrist_act_log, 'g-', linewidth=1.5, label="Actual (MPC)")
axes[2].axvline(T_raise, color='gray', linestyle=':', alpha=0.5)
axes[2].axvline(T_raise + T_wait, color='gray', linestyle=':', alpha=0.5)
axes[2].axvline(T_raise + T_wait + T_wave, color='gray', linestyle=':', alpha=0.5)
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Wrist Angle [rad]")
axes[2].set_title("Wrist Joint (arm_r_joint7) - MPC Control")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================
# Plot 2: Tracking Error
# ===============================
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

shoulder_error = shoulder_ref_log - shoulder_act_log
upperarm_error = upperarm_ref_log - upperarm_act_log
wrist_error = wrist_ref_log - wrist_act_log

axes[0].plot(time_log, shoulder_error, 'b-')
axes[0].set_ylabel("Shoulder Error [rad]")
axes[0].set_title("Tracking Error")
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_log, upperarm_error, 'm-')
axes[1].set_ylabel("Upper Arm Error [rad]")
axes[1].grid(True, alpha=0.3)

axes[2].plot(time_log, wrist_error, 'g-')
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Wrist Error [rad]")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================
# Plot 3: Applied Torques
# ===============================
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time_log, tau_shoulder_log, 'b-', linewidth=1.5)
axes[0].set_ylabel("Shoulder Torque [Nm]")
axes[0].set_title("Applied Torques (MPC)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_log, tau_upperarm_log, 'm-', linewidth=1.5)
axes[1].set_ylabel("Upper Arm Torque [Nm]")
axes[1].grid(True, alpha=0.3)

axes[2].plot(time_log, tau_wrist_log, 'g-', linewidth=1.5)
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Wrist Torque [Nm]")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================
# 통계 출력
# ===============================
print("=" * 60)
print("📊 Tracking Performance Statistics (MPC)")
print("=" * 60)

print("\n[Shoulder Joint]")
print(f"RMSE: {np.sqrt(np.mean(shoulder_error**2)):.6f} rad ({np.rad2deg(np.sqrt(np.mean(shoulder_error**2))):.3f} deg)")
print(f"Max Error: {np.max(np.abs(shoulder_error)):.6f} rad ({np.rad2deg(np.max(np.abs(shoulder_error))):.3f} deg)")
print(f"Final Error: {shoulder_error[-1]:.6f} rad ({np.rad2deg(shoulder_error[-1]):.3f} deg)")

wave_start_idx = np.argmin(np.abs(time_log - (T_raise + T_wait)))
upperarm_error_wave = upperarm_error[wave_start_idx:]
wrist_error_wave = wrist_error[wave_start_idx:]

print("\n[Upper Arm Joint - Waving Phase]")
print(f"RMSE: {np.sqrt(np.mean(upperarm_error_wave**2)):.6f} rad ({np.rad2deg(np.sqrt(np.mean(upperarm_error_wave**2))):.3f} deg)")
print(f"Max Error: {np.max(np.abs(upperarm_error_wave)):.6f} rad ({np.rad2deg(np.max(np.abs(upperarm_error_wave))):.3f} deg)")

print("\n[Wrist Joint - Waving Phase]")
print(f"RMSE: {np.sqrt(np.mean(wrist_error_wave**2)):.6f} rad ({np.rad2deg(np.sqrt(np.mean(wrist_error_wave**2))):.3f} deg)")
print(f"Max Error: {np.max(np.abs(wrist_error_wave)):.6f} rad ({np.rad2deg(np.max(np.abs(wrist_error_wave))):.3f} deg)")

print("\n[Control Effort]")
print(f"Mean |Shoulder Torque|: {np.mean(np.abs(tau_shoulder_log)):.3f} Nm")
print(f"Mean |Upper Arm Torque|: {np.mean(np.abs(tau_upperarm_log)):.3f} Nm")
print(f"Mean |Wrist Torque|: {np.mean(np.abs(tau_wrist_log)):.3f} Nm")

print("=" * 60)
"""
verify_mpc_dynamics.py

MPC의 dynamics model 정확도를 검증합니다.

실행 방법:
    python verify_mpc_dynamics.py

출력:
    - MPC 예측 오차 통계
    - 시간에 따른 오차 변화 그래프
    - 관절별 오차 분석
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataGet import (
    SimulationConfig, MPCConfig, PathConfig,
    generate_reference_trajectory,
    TorqueMPC,
    setup_robot
)


def verify_mpc_dynamics_single_step(controller, data):
    """
    단일 스텝에서 MPC 예측 vs 실제 시뮬레이션 비교
    
    Args:
        controller: TorqueMPC instance
        data: Current MuJoCo data
    
    Returns:
        error_dict: 오차 정보 딕셔너리
    """
    # 현재 상태 저장
    q = data.qpos[controller.joint_ids].copy()
    qdot = data.qvel[controller.joint_ids].copy()
    
    # 테스트할 토크 (MPC가 실제로 계산한 값 사용)
    tau_test = np.zeros(3)  # 또는 controller.compute_control_from_state()의 출력
    
    # === 1. MPC 예측 ===
    controller._cache_dynamics_from_state(data.qpos, data.qvel)
    q_next_mpc, qdot_next_mpc = controller._predict_state(q, qdot, tau_test)
    
    # === 2. 실제 시뮬레이션 ===
    data_copy = mujoco.MjData(controller.model)
    data_copy.qpos[:] = data.qpos
    data_copy.qvel[:] = data.qvel
    
    # 같은 토크 적용
    data_copy.ctrl[:] = 0.0
    for i, joint_id in enumerate(controller.joint_ids):
        # 여기서는 tau_test를 motor에 매핑해야 함
        # 간단히 하기 위해 0으로 설정 (gravity만 있는 경우 테스트)
        pass
    
    # MuJoCo 시뮬레이션 1 step
    mujoco.mj_step(controller.model, data_copy)
    
    q_next_real = data_copy.qpos[controller.joint_ids]
    qdot_next_real = data_copy.qvel[controller.joint_ids]
    
    # === 3. 오차 계산 ===
    q_error = q_next_mpc - q_next_real
    qdot_error = qdot_next_mpc - qdot_next_real
    
    return {
        'q_error': q_error,
        'qdot_error': qdot_error,
        'q_error_norm': np.linalg.norm(q_error),
        'qdot_error_norm': np.linalg.norm(qdot_error),
        'q_mpc': q_next_mpc,
        'q_real': q_next_real,
        'qdot_mpc': qdot_next_mpc,
        'qdot_real': qdot_next_real,
    }


def verify_mpc_dynamics_trajectory(model, controller, robot, duration=5.0, dt=0.005):
    """
    전체 궤적에 대해 MPC dynamics 정확도 검증
    
    Args:
        model: MuJoCo model
        controller: TorqueMPC instance
        robot: RobotInterface instance
        duration: Test duration [s]
        dt: Time step [s]
    
    Returns:
        results: 검증 결과 딕셔너리
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    n_steps = int(duration / dt)
    shoulder_id, upperarm_id, wrist_id = robot.get_qpos_ids()
    controlled_joint_ids = robot.get_controlled_joint_ids()
    
    # 로깅 변수
    time_log = []
    q_error_norm_log = []
    qdot_error_norm_log = []
    q_error_per_joint = [[] for _ in range(3)]
    qdot_error_per_joint = [[] for _ in range(3)]
    
    print("\n" + "=" * 60)
    print("🔍 MPC Dynamics Verification")
    print("=" * 60)
    print(f"Duration: {duration}s, Steps: {n_steps}")
    print(f"dt: {dt}s")
    print()
    
    for step in range(n_steps):
        t = step * dt
        
        # Reference trajectory 생성
        q_ref_dict = generate_reference_trajectory(t, shoulder_id, upperarm_id, wrist_id)
        q_ref_array = np.array([
            q_ref_dict[controlled_joint_ids[0]],
            q_ref_dict[controlled_joint_ids[1]],
            q_ref_dict[controlled_joint_ids[2]]
        ])
        
        # MPC 제어 계산
        tau_mpc, _ = controller.compute_control_from_state(
            data.qpos, data.qvel, q_ref_array
        )
        
        # Dynamics 검증 (tau_mpc 적용 전 상태에서)
        error_info = verify_mpc_dynamics_with_torque(
            controller, data, robot, tau_mpc
        )
        
        # 로깅
        time_log.append(t)
        q_error_norm_log.append(error_info['q_error_norm'])
        qdot_error_norm_log.append(error_info['qdot_error_norm'])
        
        for j in range(3):
            q_error_per_joint[j].append(error_info['q_error'][j])
            qdot_error_per_joint[j].append(error_info['qdot_error'][j])
        
        # 실제 제어 적용 및 시뮬레이션
        robot.apply_torques(data, tau_mpc)
        mujoco.mj_step(model, data)
        
        # 주기적으로 진행상황 출력
        if step % 200 == 0:
            print(f"[{t:.2f}s] q_error_norm: {error_info['q_error_norm']:.6f} rad, "
                  f"qdot_error_norm: {error_info['qdot_error_norm']:.6f} rad/s")
    
    # 통계 계산
    results = {
        'time': np.array(time_log),
        'q_error_norm': np.array(q_error_norm_log),
        'qdot_error_norm': np.array(qdot_error_norm_log),
        'q_error_per_joint': [np.array(arr) for arr in q_error_per_joint],
        'qdot_error_per_joint': [np.array(arr) for arr in qdot_error_per_joint],
    }
    
    return results


def verify_mpc_dynamics_with_torque(controller, data, robot, tau):
    """
    특정 토크를 적용했을 때의 dynamics 검증
    
    Args:
        controller: TorqueMPC instance
        data: Current MuJoCo data
        robot: RobotInterface instance
        tau: Torque to apply [3]
    
    Returns:
        error_dict: 오차 정보
    """
    # 현재 상태
    q = data.qpos[controller.joint_ids].copy()
    qdot = data.qvel[controller.joint_ids].copy()
    
    # === MPC 예측 ===
    controller._cache_dynamics_from_state(data.qpos, data.qvel)
    q_next_mpc, qdot_next_mpc = controller._predict_state(q, qdot, tau)
    
    # === 실제 시뮬레이션 ===
    data_copy = mujoco.MjData(controller.model)
    data_copy.qpos[:] = data.qpos
    data_copy.qvel[:] = data.qvel
    
    # 같은 토크 적용
    robot.apply_torques(data_copy, tau)
    mujoco.mj_step(controller.model, data_copy)
    
    q_next_real = data_copy.qpos[controller.joint_ids]
    qdot_next_real = data_copy.qvel[controller.joint_ids]
    
    # 오차 계산
    q_error = q_next_mpc - q_next_real
    qdot_error = qdot_next_mpc - qdot_next_real
    
    return {
        'q_error': q_error,
        'qdot_error': qdot_error,
        'q_error_norm': np.linalg.norm(q_error),
        'qdot_error_norm': np.linalg.norm(qdot_error),
    }


def print_statistics(results, joint_names):
    """
    검증 결과 통계 출력
    
    Args:
        results: verify_mpc_dynamics_trajectory의 출력
        joint_names: 관절 이름 리스트
    """
    print("\n" + "=" * 60)
    print("📊 MPC Dynamics Accuracy Statistics")
    print("=" * 60)
    
    # 전체 통계
    print(f"\n[Overall Statistics]")
    print(f"Mean q_error_norm:   {np.mean(results['q_error_norm']):.6f} rad")
    print(f"Max q_error_norm:    {np.max(results['q_error_norm']):.6f} rad")
    print(f"Std q_error_norm:    {np.std(results['q_error_norm']):.6f} rad")
    print()
    print(f"Mean qdot_error_norm: {np.mean(results['qdot_error_norm']):.6f} rad/s")
    print(f"Max qdot_error_norm:  {np.max(results['qdot_error_norm']):.6f} rad/s")
    print(f"Std qdot_error_norm:  {np.std(results['qdot_error_norm']):.6f} rad/s")
    
    # 관절별 통계
    print(f"\n[Per-Joint Position Error Statistics]")
    for j, name in enumerate(joint_names):
        q_err = results['q_error_per_joint'][j]
        print(f"{name:20s}: mean={np.mean(np.abs(q_err)):.6f} rad, "
              f"max={np.max(np.abs(q_err)):.6f} rad, "
              f"std={np.std(q_err):.6f} rad")
    
    print(f"\n[Per-Joint Velocity Error Statistics]")
    for j, name in enumerate(joint_names):
        qdot_err = results['qdot_error_per_joint'][j]
        print(f"{name:20s}: mean={np.mean(np.abs(qdot_err)):.6f} rad/s, "
              f"max={np.max(np.abs(qdot_err)):.6f} rad/s, "
              f"std={np.std(qdot_err):.6f} rad/s")
    
    print("=" * 60)
    
    # 해석 가이드
    print("\n💡 Interpretation Guide:")
    print("=" * 60)
    if np.mean(results['q_error_norm']) < 1e-6:
        print("✅ EXCELLENT: MPC dynamics model is highly accurate")
    elif np.mean(results['q_error_norm']) < 1e-4:
        print("✅ GOOD: MPC dynamics model is reasonably accurate")
    elif np.mean(results['q_error_norm']) < 1e-3:
        print("⚠️  WARNING: MPC dynamics model has noticeable errors")
    else:
        print("🚨 CRITICAL: MPC dynamics model is significantly inaccurate!")
        print("   → This explains why residual torques are so large")
        print("   → Consider fixing the dynamics model or using a better integrator")
    print("=" * 60)


def plot_results(results, joint_names):
    """
    검증 결과 시각화
    
    Args:
        results: verify_mpc_dynamics_trajectory의 출력
        joint_names: 관절 이름 리스트
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Row 1: Overall error norms
    axes[0, 0].plot(results['time'], results['q_error_norm'], 'b-', linewidth=1.5)
    axes[0, 0].set_ylabel('Position Error Norm [rad]')
    axes[0, 0].set_title('MPC Prediction Error - Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(results['time'], results['qdot_error_norm'], 'r-', linewidth=1.5)
    axes[0, 1].set_ylabel('Velocity Error Norm [rad/s]')
    axes[0, 1].set_title('MPC Prediction Error - Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Per-joint position errors
    colors = ['b', 'm', 'g']
    for j, (name, color) in enumerate(zip(joint_names, colors)):
        axes[1, 0].plot(results['time'], results['q_error_per_joint'][j], 
                       color=color, linewidth=1.5, label=name, alpha=0.7)
    axes[1, 0].set_ylabel('Position Error [rad]')
    axes[1, 0].set_title('Position Error by Joint')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Row 2: Per-joint velocity errors
    for j, (name, color) in enumerate(zip(joint_names, colors)):
        axes[1, 1].plot(results['time'], results['qdot_error_per_joint'][j], 
                       color=color, linewidth=1.5, label=name, alpha=0.7)
    axes[1, 1].set_ylabel('Velocity Error [rad/s]')
    axes[1, 1].set_title('Velocity Error by Joint')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Row 3: Error distributions (histograms)
    all_q_errors = np.concatenate(results['q_error_per_joint'])
    axes[2, 0].hist(all_q_errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[2, 0].set_xlabel('Position Error [rad]')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Position Error Distribution')
    axes[2, 0].grid(True, alpha=0.3)
    
    all_qdot_errors = np.concatenate(results['qdot_error_per_joint'])
    axes[2, 1].hist(all_qdot_errors, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[2, 1].set_xlabel('Velocity Error [rad/s]')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].set_title('Velocity Error Distribution')
    axes[2, 1].grid(True, alpha=0.3)
    
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel('Time [s]')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main verification function"""
    
    print("\n" + "=" * 60)
    print("🔍 MPC Dynamics Model Verification")
    print("=" * 60)
    
    # Load model
    paths = PathConfig.get_paths()
    xml_path = paths['xml_path']
    
    print(f"\n📁 Loading model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    print("✅ Model loaded")
    
    # Setup robot
    robot = setup_robot(model)
    controlled_joint_ids = robot.get_controlled_joint_ids()
    joint_names = robot.get_joint_names()
    
    # Create MPC controller
    sim_cfg = SimulationConfig
    mpc_cfg = MPCConfig
    
    controller = TorqueMPC(
        model=model,
        joint_ids=controlled_joint_ids,
        horizon=mpc_cfg.HORIZON,
        dt=sim_cfg.SIM_DT
    )
    
    print(f"\n🎮 MPC Controller:")
    print(f"   Horizon: {mpc_cfg.HORIZON}")
    print(f"   dt: {sim_cfg.SIM_DT}s")
    
    # Run verification
    results = verify_mpc_dynamics_trajectory(
        model, controller, robot, 
        duration=5.0,  # 5초 테스트
        dt=sim_cfg.SIM_DT
    )
    
    # Print statistics
    print_statistics(results, joint_names)
    
    # Plot results
    print("\n📊 Generating plots...")
    plot_results(results, joint_names)
    
    print("\n" + "=" * 60)
    print("✅ Verification completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
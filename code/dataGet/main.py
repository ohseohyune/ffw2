"""
Main Script for MPC + Residual Torque Dataset Generation

모든 모듈을 조합하여 MPC 시뮬레이션을 실행하고 데이터셋을 생성합니다.

실행 방법:
    python main.py

출력:
    - delta_tau_dataset.npz: 학습용 데이터셋
    - 제어 성능 그래프들
"""

import numpy as np
import mujoco
import mujoco.viewer
import time

from eval.evaluation import PerformanceEvaluator 

# Import all modules
from .config import (
    SimulationConfig, MPCConfig, PathConfig, DatasetConfig
)
from .trajectory import generate_reference_trajectory, get_trajectory_phases
from .mpc_controller import TorqueMPC
from .async_utils import MPCAsyncManager, MPCInput
from .data_logger import DatasetCollector, TrackingLogger, TorqueComparisonLogger
from .residual_calculator import ResidualCalculator
from .robot_setup import setup_robot
from .visualization import (
    plot_all_results, print_residual_analysis,
    plot_six_torques_comparison, plot_all_six_torques, plot_residual_accuracy
)


def main():
    """Main simulation function"""
    
    print("\n" + "=" * 60)
    print("🚀 MPC + Residual Torque Dataset Generation")
    print("=" * 60)
    
    # ===============================
    # 1. Load MuJoCo Model
    # ===============================
    paths = PathConfig.get_paths()
    xml_path = paths['xml_path']
    
    print(f"\n📁 Loading model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print("✅ Model loaded successfully")
    
    # ===============================
    # 2. Setup Robot Interface
    # ===============================
    robot = setup_robot(model)
    controlled_joint_ids = robot.get_controlled_joint_ids()
    motor_ids = robot.get_motor_ids()
    # shoulder_id, upperarm_id, wrist_id = robot.get_qpos_ids()
    shoulder_id = robot.get_qpos_ids()
    joint_names = robot.get_joint_names()
    
    # ===============================
    # 3. Setup Simulation Parameters
    # ===============================
    sim_cfg = SimulationConfig
    dataset_cfg = DatasetConfig
    
    sim_dt = sim_cfg.SIM_DT
    mpc_rate_hz = sim_cfg.MPC_RATE_HZ
    sim_duration = sim_cfg.SIM_DURATION
    realtime_factor = sim_cfg.REALTIME_FACTOR
    
    push_interval_steps = dataset_cfg.get_push_interval_steps()
    n_steps = dataset_cfg.get_total_steps()
    
    print(f"\n⚙️  Simulation Settings:")
    print(f"   Simulation dt: {sim_dt*1000:.1f} ms ({1/sim_dt:.0f} Hz)")
    print(f"   MPC rate: {mpc_rate_hz:.0f} Hz")
    print(f"   Duration: {sim_duration:.1f} s")
    print(f"   Total steps: {n_steps}")
    print(f"   MPC input push interval: {push_interval_steps} steps")
    
    # ===============================
    # 4. Create MPC Controller
    # ===============================
    mpc_cfg = MPCConfig
    controller = TorqueMPC(
        model=model,
        joint_ids=controlled_joint_ids,
        horizon=mpc_cfg.HORIZON,
        dt=sim_dt
    )
    
    print(f"\n🎮 MPC Controller:")
    print(f"   Horizon: {mpc_cfg.HORIZON}")
    print(f"   Max iterations: {mpc_cfg.MAX_ITER}")
    
    # ===============================
    # 5. Setup Async MPC Manager
    # ===============================
    mpc_manager = MPCAsyncManager(model, controller, mpc_rate_hz)
    
    # ===============================
    # 6. Setup Data Loggers
    # ===============================
    dataset_collector = DatasetCollector()
    tracking_logger = TrackingLogger()
    torque_comparison_logger = TorqueComparisonLogger()  # NEW: 6가지 토크 비교용
    residual_calc = ResidualCalculator(model, controlled_joint_ids)
    
    # ===============================
    # 7. Initialize Simulation
    # ===============================
    print("\n🎬 Initializing simulation...")
    
    # Initial gravity compensation
    mujoco.mj_forward(model, data)
    
    # Initialize tau_hold for controlled joints only
    tau_hold = np.zeros(len(controlled_joint_ids))
    tau_mpc_hold = np.zeros(len(controlled_joint_ids))
    
    # First MPC input
    # q_ref_dict = generate_reference_trajectory(0.0, shoulder_id, upperarm_id, wrist_id)
    # q_ref_dict = generate_reference_trajectory(0.0, shoulder_id)
    # q_ref_array = np.array([
    #     q_ref_dict[controlled_joint_ids[0]]
    #     # q_ref_dict[controlled_joint_ids[1]],
    #     # q_ref_dict[controlled_joint_ids[2]]
    # ])
    # t0 = 0.0
    # q_ref = generate_reference_trajectory(t0, shoulder_id)  # float
    # print("qref Type: ")
    # print(type(q_ref), q_ref)
    # q_ref_array = np.array([q_ref])

    # Initial reference (always ndarray)
    t0 = 0.0
    q_ref_dict = generate_reference_trajectory(t0, shoulder_id)
    q_ref_array = np.array([q_ref_dict[shoulder_id]])

    # Track previous reference for velocity estimation
    q_ref_prev = q_ref_array.copy()
    
    # Start MPC thread
    mpc_manager.start()
    mpc_manager.push_input(
        q=data.qpos.copy(),
        qdot=data.qvel.copy(),
        q_ref=q_ref_array.copy(),
        q_ref_prev=q_ref_prev.copy(),
        stamp=time.time()
    )
    
    # ===============================
    # 8. Run Simulation
    # ===============================
    print("\n▶️  Starting simulation with viewer...")
    print(f"   Mode: Async MPC (sim: {1/sim_dt:.0f} Hz, MPC: {mpc_rate_hz:.0f} Hz)")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_steps):
            t = step * sim_dt
            
            # Generate reference trajectory
            # q_ref_dict = generate_reference_trajectory(t, shoulder_id, upperarm_id, wrist_id)
            # q_ref_dict = generate_reference_trajectory(t, shoulder_id)
            # q_ref_array = np.array([
            #     q_ref_dict[controlled_joint_ids[0]]
            #     # q_ref_dict[controlled_joint_ids[1]],
            #     # q_ref_dmpc_manager.push_inputct[controlled_joint_ids[2]]
            # ])
                # Generate reference and always use ndarray
            q_ref_dict = generate_reference_trajectory(t, shoulder_id)
            q_ref_array = np.array([q_ref_dict[shoulder_id]])  
                    
            
            # Push input to MPC thread periodically
            if step % max(push_interval_steps, 1) == 0:
                mpc_manager.push_input(
                    q=data.qpos.copy(),
                    qdot=data.qvel.copy(),
                    q_ref=q_ref_array.copy(),
                    q_ref_prev=q_ref_prev.copy(),
                    stamp=time.time()
                )
            
            # Update previous reference for next iteration
            q_ref_prev = q_ref_array.copy()
            
            # Read latest torque from MPC thread (non-blocking)
            ok, tau_new, tau_mpc_new, _ = mpc_manager.read_torque()
            if ok:
                tau_hold = tau_new
                tau_mpc_hold = tau_mpc_new
            
            # Get current state for dataset (before step)
            q_k = data.qpos.copy()
            qdot_k = data.qvel.copy()
            
            # Apply control using motor actuators
            robot.apply_torques(data, tau_hold)

            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Calculate residual torque (after step)
            tau_residual = residual_calc.compute_residual(
                data, q_k, qdot_k, tau_mpc_hold, sim_dt
            )
            
            # Calculate true required torque 
            tau_true = tau_mpc_hold + tau_residual
            
            # Store dataset sample
            dataset_collector.add_sample(
                q=q_k[controlled_joint_ids],
                qdot=qdot_k[controlled_joint_ids],
                tau_mpc=tau_mpc_hold,
                delta_tau=tau_residual
            )
            
            # Store 6-torque comparison (for dataGet phase, delta_trained = 0)
            torque_comparison_logger.add_sample(
                t=t,
                tau_mpc=tau_mpc_hold,
                tau_true=tau_true,
                # delta_trained=np.zeros(3),  # No NN in dataGet phase
                delta_trained = np.zeros(len(controlled_joint_ids)),
                tau_final=tau_hold  # Same as tau_mpc in this phase
            )
            
            # Store tracking data
            tracking_logger.add_sample(
                t=t,
                q_ref_dict=q_ref_dict,
                q_act=data.qpos,
                tau=tau_mpc_hold,
                shoulder_id=shoulder_id
                # upperarm_id=upperarm_id,
                # wrist_id=wrist_id
            )
            
            # Real-time control (50% speed)
            time.sleep(sim_dt * realtime_factor)
    
    # ===============================
    # 9. Stop MPC Thread
    # ===============================
    print("\n⏹️  Stopping MPC thread...")
    mpc_manager.stop()
    print("✅ Simulation finished")
    
    # ===============================
    # 10. Save Dataset
    # ===============================
    print("\n💾 Saving dataset...")
    dataset_path = dataset_collector.save_dataset()
    
    # ===============================
    # 11. Analyze Results
    # ===============================
    print("\n📊 Analyzing results...")
    
    # Residual torque analysis
    stats = dataset_collector.get_statistics()
    print_residual_analysis(stats, joint_names)
    
    # Tracking performance statistics
    phases = get_trajectory_phases()
    wave_start_time = phases['wait']
    tracking_logger.print_statistics(joint_names, wave_start_time)
    
    # ===============================
    # 12. Visualization
    # ===============================
    plot_all_results(tracking_logger, dataset_collector, joint_names)
    
    # NEW: 6가지 토크 비교 그래프
    print("\n📊 Generating 6-torque comparison plots...")
    plot_six_torques_comparison(torque_comparison_logger, joint_names)
    plot_all_six_torques(torque_comparison_logger, joint_names)
    
    print("\n" + "=" * 60)
    print("✅ All tasks completed successfully!")
    print("=" * 60)
    print(f"\n📦 Output files:")
    print(f"   - Dataset: {dataset_path}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Train residual NN: python train_residual_nn.py")
    print(f"   2. Apply trained model: python apply_nn.py")
    print("=" * 60 + "\n")

    evaluator = PerformanceEvaluator(dt=0.005)
    result_mpc = evaluator.evaluate(tracking_logger,label="MPC only")
    
    np.savez(
        "result_mpc_only.npz",
        **result_mpc
    )

if __name__ == "__main__":
    main()
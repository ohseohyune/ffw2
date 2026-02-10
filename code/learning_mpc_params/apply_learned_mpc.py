"""
Apply Learned MPC Weights

역최적제어로 학습한 MPC 가중치를 적용하여 시뮬레이션을 실행하고
원래 가중치와 성능을 비교합니다.

실행 순서:
    1. python main.py               (원래 가중치로 시연 데이터 생성)
    2. python learn_mpc_weights.py  (가중치 학습)
    3. python apply_learned_mpc.py  (이 스크립트 - 학습된 가중치로 실행)
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os

sys.path.append('/home/seohy/colcon_ws/src/ffw2/code')

from dataGet.config import (
    SimulationConfig, MPCConfig, PathConfig, DatasetConfig
)
from dataGet.trajectory import generate_reference_trajectory
from dataGet.mpc_controller import TorqueMPC
from dataGet.async_utils import MPCAsyncManager
from dataGet.data_logger import TrackingLogger
from dataGet.robot_setup import setup_robot
from eval.evaluation import PerformanceEvaluator
from .inverse_optimal_control import apply_learned_weights_to_mpc


def main():
    print("\n" + "="*80)
    print("🚀 Testing MPC with Learned Cost Weights")
    print("="*80)
    
    # ===============================
    # 1. Load Learned Weights
    # ===============================
    weights_path = '/home/seohy/colcon_ws/src/ffw2/code/learning_mpc_params/learned_mpc_weights.npz'
    
    if not os.path.exists(weights_path):
        print(f"\n❌ Error: Learned weights not found at {weights_path}")
        print("   Please run learn_mpc_weights.py first")
        return
    
    print(f"\n📊 Loading learned weights: {weights_path}")
    weights_data = np.load(weights_path, allow_pickle=True)
    
    theta_learned = weights_data['theta_learned']
    theta_init = weights_data['theta_init']
    
    print(f"\n✅ Loaded weights:")
    print(f"   Q_pos: {theta_learned[0]:.2f}  (original: {theta_init[0]:.2f})")
    print(f"   Q_vel: {theta_learned[1]:.2f}  (original: {theta_init[1]:.2f})")
    print(f"   R_tau: {theta_learned[2]:.6f}  (original: {theta_init[2]:.6f})")
    print(f"   Q_terminal: {theta_learned[3]:.2f}  (original: {theta_init[3]:.2f})")
    
    # ===============================
    # 2. Setup MuJoCo
    # ===============================
    paths = PathConfig.get_paths()
    xml_path = paths['xml_path']
    
    print(f"\n📁 Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    robot = setup_robot(model)
    controlled_joint_ids = robot.get_controlled_joint_ids()
    shoulder_id = robot.get_qpos_ids()
    joint_names = robot.get_joint_names()
    
    # ===============================
    # 3. Create MPC with Learned Weights
    # ===============================
    controller = TorqueMPC(
        model=model,
        joint_ids=controlled_joint_ids,
        horizon=MPCConfig.HORIZON,
        dt=SimulationConfig.SIM_DT
    )
    
    # 학습된 가중치 적용
    apply_learned_weights_to_mpc(controller, theta_learned)
    
    # ===============================
    # 4. Setup Simulation
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
    print(f"   Duration: {sim_duration:.1f} s")
    print(f"   MPC rate: {mpc_rate_hz:.0f} Hz")
    print(f"   Total steps: {n_steps}")
    
    # ===============================
    # 5. Setup Async MPC
    # ===============================
    mpc_manager = MPCAsyncManager(model, controller, mpc_rate_hz)
    tracking_logger = TrackingLogger()
    
    # ===============================
    # 6. Initialize
    # ===============================
    print("\n🎬 Initializing simulation...")
    mujoco.mj_forward(model, data)
    
    tau_hold = np.zeros(len(controlled_joint_ids))
    tau_mpc_hold = np.zeros(len(controlled_joint_ids))
    
    # Initial reference
    t0 = 0.0
    q_ref_dict = generate_reference_trajectory(t0, shoulder_id)
    q_ref_array = np.array([q_ref_dict[shoulder_id]])
    q_ref_prev = q_ref_array.copy()
    
    # Start MPC
    mpc_manager.start()
    mpc_manager.push_input(
        q=data.qpos.copy(),
        qdot=data.qvel.copy(),
        q_ref=q_ref_array.copy(),
        q_ref_prev=q_ref_prev.copy(),
        stamp=time.time()
    )
    
    # ===============================
    # 7. Run Simulation
    # ===============================
    print("\n▶️  Running simulation with LEARNED weights...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_steps):
            t = step * sim_dt
            
            # Reference trajectory
            q_ref_dict = generate_reference_trajectory(t, shoulder_id)
            q_ref_array = np.array([q_ref_dict[shoulder_id]])
            
            # Push MPC input
            if step % max(push_interval_steps, 1) == 0:
                mpc_manager.push_input(
                    q=data.qpos.copy(),
                    qdot=data.qvel.copy(),
                    q_ref=q_ref_array.copy(),
                    q_ref_prev=q_ref_prev.copy(),
                    stamp=time.time()
                )
            
            q_ref_prev = q_ref_array.copy()
            
            # Read torque
            ok, tau_new, tau_mpc_new, _ = mpc_manager.read_torque()
            if ok:
                tau_hold = tau_new
                tau_mpc_hold = tau_mpc_new
            
            # Apply control
            robot.apply_torques(data, tau_hold)
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Log data
            tracking_logger.add_sample(
                t=t,
                q_ref_dict=q_ref_dict,
                q_act=data.qpos,
                tau=tau_mpc_hold,
                shoulder_id=shoulder_id
            )
            
            # Real-time control
            time.sleep(sim_dt * realtime_factor)
    
    # ===============================
    # 8. Stop MPC
    # ===============================
    mpc_manager.stop()
    print("\n✅ Simulation finished")
    
    # ===============================
    # 9. Evaluate Performance
    # ===============================
    print("\n📊 Evaluating performance with learned weights...")
    
    evaluator = PerformanceEvaluator(dt=sim_dt)
    result_learned = evaluator.evaluate(tracking_logger, label="MPC (Learned)")
    
    # Load original results for comparison
    original_results_path = "result_mpc_only.npz"
    
    if os.path.exists(original_results_path):
        print(f"\n📊 Loading original results: {original_results_path}")
        original_data = np.load(original_results_path, allow_pickle=True)
        
        # Reconstruct result dict
        result_original = {key: original_data[key].item() for key in original_data.keys()}
        
        # Compare
        print("\n" + "="*80)
        print("🔍 Performance Comparison: Original vs Learned Weights")
        print("="*80)
        evaluator.compare(result_original, result_learned)
    else:
        print(f"\n⚠️  Original results not found: {original_results_path}")
        print("   Run main.py first to generate baseline results")
        
        # Print learned results only
        print("\n" + "="*80)
        print("📊 Performance with Learned Weights")
        print("="*80)
        for key, value in result_learned.items():
            if key != 'label':
                print(f"   {key:25s}: {value:.6f}")
    
    # ===============================
    # 10. Save Results
    # ===============================
    save_path = "result_mpc_learned.npz"
    np.savez(save_path, **result_learned)
    print(f"\n💾 Saved results: {save_path}")
    
    # ===============================
    # 11. Summary
    # ===============================
    print("\n" + "="*80)
    print("✅ Evaluation Completed!")
    print("="*80)
    
    print(f"\n📋 Key Metrics (Learned Weights):")
    print(f"   RMSE (all):        {result_learned['rmse_all']:.6f} rad")
    print(f"   RMSE (transition): {result_learned['rmse_transition']:.6f} rad")
    print(f"   RMSE (steady):     {result_learned['rmse_steady']:.6f} rad")
    print(f"   Rise time:         {result_learned['rise_time']:.3f} s")
    print(f"   Settling time:     {result_learned['settling_time']:.3f} s")
    print(f"   Overshoot:         {result_learned['overshoot']:.2f} %")
    print(f"   Mean |tau|:        {result_learned['mean_abs_tau']:.2f} Nm")
    print(f"   Control effort:    {result_learned['control_effort']:.2f} Nm²s")
    
    print(f"\n📦 Output Files:")
    print(f"   - {save_path}")
    
    if os.path.exists(original_results_path):
        # Calculate improvement
        improvements = {}
        for key in ['rmse_all', 'rmse_transition', 'rmse_steady', 
                   'steady_state_error', 'mean_abs_tau']:
            orig = result_original[key]
            learned = result_learned[key]
            improvement = ((orig - learned) / orig) * 100
            improvements[key] = improvement
        
        print(f"\n🎯 Improvements over Original:")
        for key, imp in improvements.items():
            arrow = "✅" if imp > 0 else "❌"
            print(f"   {arrow} {key:25s}: {imp:+6.2f}%")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
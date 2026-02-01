"""
applyNNtoMPC.py

Role:
    Load trained residual NN (residual_nn.pt) and apply it to MPC torque.
    τ_total = τ_mpc + α · clamp(Δτ_hat, -max, +max)
    
Input:
    - Trained model: residual_nn.pt (from TrainNN)
    - Current state: q, qdot, τ_mpc (from MPC controller)
    
Output:
    - τ_total: Final torque to apply to robot
    
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import torch
import torch.nn as nn
import threading
from dataclasses import dataclass

from eval.evaluation import PerformanceEvaluator 

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataGet import (
    SimulationConfig, MPCConfig, PathConfig, TrajectoryConfig,
    generate_reference_trajectory, get_trajectory_phases,
    TorqueMPC,
    MPCInput, SharedTorqueBuffer, SharedMPCInput,
    TrackingLogger, TorqueComparisonLogger,
    setup_robot,
    plot_joint_tracking, plot_tracking_errors, plot_applied_torques,
    plot_six_torques_comparison, plot_all_six_torques, plot_residual_accuracy
)

# Import residual calculator for computing true torque
from dataGet.residual_calculator import ResidualCalculator


# ========================================
# 1. Residual NN Model (same as TrainNN)
# ========================================
class ResidualTorqueNN(nn.Module):
    """
    Residual Torque Neural Network
    
    Input: [q(3), qdot(3), tau_mpc(3)] = 9D
    Output: [delta_tau(3)] = 3D
    """
    def __init__(self, delta_tau_max):
        super().__init__()
        self.delta_tau_max = float(delta_tau_max)
        
        # self.net = nn.Sequential(
        #     nn.Linear(9, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 3),
        #     nn.Tanh()  # Output in [-1, 1]
        # )
        
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, 9) or (9,)
        Returns:
            delta_tau: Residual torque (batch_size, 3) or (3,)
        """
        return self.delta_tau_max * self.net(x)


# ========================================
# 2. Residual Compensator
# ========================================
class ResidualCompensator:
    """
    Apply trained residual NN to MPC torque
    
    τ_total = τ_mpc + α · clamp(Δτ_hat, -max, +max)
    """
    
    def __init__(self, model_path: str, joint_ids: list, 
                 alpha: float, delta_tau_max: float):
        """
        Args:
            model_path: Path to residual_nn.pt
            joint_ids: List of controlled joint indices [shoulder, upperarm, wrist]
            alpha: Residual gain (default 1.0)
            delta_tau_max: Clamp limit [Nm] (default 10.0)
        """
        self.joint_ids = joint_ids
        self.n_joints = len(joint_ids)
        self.alpha = alpha
        self.delta_tau_max = delta_tau_max
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model = ResidualTorqueNN(delta_tau_max=checkpoint["delta_tau_max"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"[ResidualCompensator] Loaded model from {model_path}")
        print(f"  Controlled joints: {joint_ids}")
        print(f"  Residual gain α: {alpha}")
        print(f"  Clamp limit: ±{delta_tau_max} Nm")

    def compute_residual(self, q_full: np.ndarray, qdot_full: np.ndarray, 
                        tau_mpc: np.ndarray) -> np.ndarray:
        """
        Compute residual torque using trained NN
        
        Args:
            q_full: Full joint positions
            qdot_full: Full joint velocities
            tau_mpc: MPC torque for controlled joints (3,)
        
        Returns:
            delta_tau: Residual torque for controlled joints (3,)
        """
        # Extract controlled joint states (following dataGet structure)
        q_sel = q_full[self.joint_ids]      # (3,)
        qdot_sel = qdot_full[self.joint_ids]  # (3,)
        
        # Construct input: [q(3), qdot(3), tau_mpc(3)] = 9D
        x = np.concatenate([q_sel, qdot_sel, tau_mpc])
        
        # Safety check
        if not np.all(np.isfinite(x)):
            print("[WARN] Invalid NN input detected, returning zero residual")
            return np.zeros(self.n_joints)
        
        # NN inference
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float()
            delta_tau_hat = self.model(x_tensor).numpy()  # (3,)
        
        # Safety check for NN output
        if not np.all(np.isfinite(delta_tau_hat)):
            print("[WARN] NN output NaN/Inf detected, returning zero residual")
            return np.zeros(self.n_joints)
        
        # Apply gain and clamp
        delta_tau = self.alpha * np.clip(delta_tau_hat, 
                                         -self.delta_tau_max, 
                                         self.delta_tau_max)
        
        return delta_tau


# ========================================
# 3. MPC Worker with Residual Compensation
# ========================================
def mpc_worker_with_residual(model, controller: TorqueMPC,
                             residual_comp: ResidualCompensator,
                             shared_inp: SharedMPCInput,
                             shared_tau: SharedTorqueBuffer,
                             stop_event: threading.Event,
                             mpc_rate_hz: float):
    """
    MPC thread with residual NN compensation
    
    Flow:
        1. Read current state from shared_inp
        2. Compute τ_mpc using MPC controller
        3. Compute Δτ_hat using residual NN
        4. Combine: τ_total = τ_mpc + Δτ_hat
        5. Write τ_total to shared_tau
    """
    period = 1.0 / max(mpc_rate_hz, 1e-6)
    next_time = time.time()
    iter_count = 0  # iteration counter for periodic debugging prints

    while not stop_event.is_set():
        now = time.time()
        
        if now < next_time:
            time.sleep(min(0.001, next_time - now))
            continue
        
        next_time += period

        # Read latest input
        inp = shared_inp.read_latest()
        if inp is None:
            continue

        t0 = time.time()
        try:
            # Step 1: MPC optimization (with velocity reference tracking)
            tau_mpc, _ = controller.compute_control_from_state(
                inp.q, inp.qdot, inp.q_ref, inp.q_ref_prev
            )
            
            # Step 2: Residual NN compensation
            delta_tau = residual_comp.compute_residual(
                inp.q, inp.qdot, tau_mpc
            )
            
            # Step 3: Combine torques
            tau_total = tau_mpc + delta_tau
            
            # Clip to limits
            tau_total = np.clip(tau_total, 
                              controller.tau_min, 
                              controller.tau_max)
            
            # Print every 20 iterations for debugging/monitoring
            if iter_count % 20 == 0:
                tau_mpc_str = np.round(tau_mpc, 3)
                delta_str = np.round(delta_tau, 3)
                tau_total_str = np.round(tau_total, 3)
                print(f"[MPC+NN] tau_mpc: {tau_mpc_str}, delta_tau: {delta_str}, tau_total: {tau_total_str}")
            
            # Write result
            shared_tau.write(tau_total, tau_mpc, stamp=time.time())
            
            solve_time = (time.time() - t0) * 1000
            # print(f"[MPC+NN] solve: {solve_time:.1f}ms | Δτ_rms: {np.sqrt(np.mean(delta_tau**2)):.3f} Nm")
            
        except Exception as e:
            print(f"[MPC+NN] Failed: {e}")


# ========================================
# 4. Async Manager with Residual NN
# ========================================
class MPCAsyncManagerWithNN:
    """
    Async MPC manager with residual NN compensation
    """
    
    def __init__(self, model, controller, residual_comp, mpc_rate_hz):
        """
        Args:
            model: MuJoCo model
            controller: TorqueMPC instance
            residual_comp: ResidualCompensator instance
            mpc_rate_hz: MPC execution frequency [Hz]
        """
        self.model = model
        self.controller = controller
        self.residual_comp = residual_comp
        self.mpc_rate_hz = mpc_rate_hz
        
        # Create shared buffers
        self.shared_inp = SharedMPCInput()
        self.shared_tau = SharedTorqueBuffer(nq=controller.nq)
        self.stop_event = threading.Event()
        
        self.mpc_thread = None

    def start(self):
        """Start MPC worker thread with residual NN"""
        if self.mpc_thread is not None and self.mpc_thread.is_alive():
            print("[MPCAsyncManagerWithNN] Thread already running")
            return
        
        self.stop_event.clear()
        self.mpc_thread = threading.Thread(
            target=mpc_worker_with_residual,
            args=(
                self.model,
                self.controller,
                self.residual_comp,
                self.shared_inp,
                self.shared_tau,
                self.stop_event,
                self.mpc_rate_hz
            ),
            daemon=True
        )
        self.mpc_thread.start()
        print(f"[MPCAsyncManagerWithNN] Started MPC+NN thread at {self.mpc_rate_hz} Hz")

    def stop(self, timeout=1.0):
        """Stop MPC worker thread"""
        if self.mpc_thread is None:
            return
        
        self.stop_event.set()
        self.mpc_thread.join(timeout=timeout)
        print("[MPCAsyncManagerWithNN] Stopped MPC+NN thread")

    def push_input(self, q, qdot, q_ref, q_ref_prev, stamp):
        """
        Push new input to MPC thread
        
        Args:
            q: Full joint positions
            qdot: Full joint velocities
            q_ref: Reference positions for controlled joints
            q_ref_prev: Previous reference positions (for velocity estimation)
            stamp: Timestamp
        """
        self.shared_inp.write(MPCInput(
            q=q.copy(),
            qdot=qdot.copy(),
            q_ref=q_ref.copy(),
            q_ref_prev=q_ref_prev.copy(),
            stamp=stamp
        ))

    def read_torque(self):
        """Read latest torque from MPC thread"""
        return self.shared_tau.read_latest()


# ========================================
# 5. Main Simulation
# ========================================
def main():
    """Main simulation with MPC + Residual NN"""
    
    print("\n" + "=" * 60)
    print("🚀 MPC + Residual NN Control")
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
    # shoulder_id, upperarm_id, wrist_id = robot.get_qpos_ids()
    shoulder_id = robot.get_qpos_ids()
    joint_names = robot.get_joint_names()
    
    # ===============================
    # 3. Setup Simulation Parameters
    # ===============================
    sim_cfg = SimulationConfig
    sim_dt = sim_cfg.SIM_DT
    mpc_rate_hz = sim_cfg.MPC_RATE_HZ
    sim_duration = sim_cfg.SIM_DURATION
    realtime_factor = sim_cfg.REALTIME_FACTOR
    
    push_interval_steps = int((1.0 / mpc_rate_hz) / sim_dt)
    n_steps = int(sim_duration / sim_dt)
    
    print(f"\n⚙️  Simulation Settings:")
    print(f"   Simulation dt: {sim_dt*1000:.1f} ms ({1/sim_dt:.0f} Hz)")
    print(f"   MPC rate: {mpc_rate_hz:.0f} Hz")
    print(f"   Duration: {sim_duration:.1f} s")
    print(f"   Total steps: {n_steps}")
    
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
    # 5. Create Residual Compensator
    # ===============================
    model_path = "/home/seohy/colcon_ws/src/olaf/ffw/code/TrainNN/residual_nn.pt"
    residual_comp = ResidualCompensator(
        model_path=model_path,
        joint_ids=controlled_joint_ids,
        alpha=2.0,  # Residual gain (can be tuned)
        delta_tau_max=50.0  # Clamp limit
    )
    
    # ===============================
    # 6. Setup Async MPC Manager with NN
    # ===============================
    mpc_manager = MPCAsyncManagerWithNN(
        model, controller, residual_comp, mpc_rate_hz
    )
    
    # ===============================
    # 7. Setup Data Loggers
    # ===============================
    tracking_logger = TrackingLogger()
    torque_comparison_logger = TorqueComparisonLogger()  # NEW: 6가지 토크 비교용
    residual_calc = ResidualCalculator(model, controlled_joint_ids)  # 실제 필요 토크 계산용
    
    # ===============================
    # 8. Initialize Simulation
    # ===============================
    print("\n🎬 Initializing simulation...")
    
    mujoco.mj_forward(model, data)
    
    tau_hold = np.zeros(len(controlled_joint_ids))
    tau_mpc_hold = np.zeros(len(controlled_joint_ids))
    
    # First MPC input
    
    # q_ref_dict = generate_reference_trajectory(0.0, shoulder_id, upperarm_id, wrist_id)
    # q_ref_array = np.array([
    #     q_ref_dict[controlled_joint_ids[0]],
    #     q_ref_dict[controlled_joint_ids[1]],
    #     q_ref_dict[controlled_joint_ids[2]]
    # ])
    t0 = 0.0
    q_ref_dict = generate_reference_trajectory(t0, shoulder_id)
    q_ref_array = np.array([q_ref_dict[shoulder_id]]) 
    
    # Track previous reference for velocity estimation
    q_ref_prev = q_ref_array.copy()
    
    # Start MPC+NN thread
    mpc_manager.start()
    mpc_manager.push_input(
        q=data.qpos.copy(),
        qdot=data.qvel.copy(),
        q_ref=q_ref_array.copy(),
        q_ref_prev=q_ref_prev.copy(),
        stamp=time.time()
    )
    
    # ===============================
    # 9. Run Simulation
    # ===============================
    print("\n▶️  Starting simulation with viewer...")
    print(f"   Mode: MPC + Residual NN (sim: {1/sim_dt:.0f} Hz, MPC: {mpc_rate_hz:.0f} Hz)")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(n_steps):
            t = step * sim_dt
            
            # Generate reference trajectory
            # q_ref_dict = generate_reference_trajectory(t, shoulder_id, upperarm_id, wrist_id)
            # q_ref_array = np.array([
            #     q_ref_dict[controlled_joint_ids[0]],
            #     q_ref_dict[controlled_joint_ids[1]],
            #     q_ref_dict[controlled_joint_ids[2]]
            # ])

            q_ref_dict = generate_reference_trajectory(t, shoulder_id)
            q_ref_array = np.array([q_ref_dict[shoulder_id]]) 
            
            
            # Push input to MPC+NN thread periodically
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
            
            # Get current state BEFORE applying control
            q_k = data.qpos.copy()
            qdot_k = data.qvel.copy()
            
            # Read latest torque (MPC + NN)
            ok, tau_new, tau_mpc_new, _ = mpc_manager.read_torque()
            if ok:
                tau_hold = tau_new
                tau_mpc_hold = tau_mpc_new
            
            # Apply control
            robot.apply_torques(data, tau_hold)
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Calculate residual and true torque (AFTER step)
            tau_residual = residual_calc.compute_residual(
                data, q_k, qdot_k, tau_mpc_hold, sim_dt
            )
            tau_true = tau_mpc_hold + tau_residual

            
            
            # Calculate NN's prediction
            delta_trained = residual_comp.compute_residual(
                q_k, qdot_k, tau_mpc_hold
            )

            # # Print torques every 20 steps
            # if step % 20 == 0:
            #     # Round for compact display
            #     tau_mpc_str = np.round(tau_mpc_hold, 3)
            #     delta_str = np.round(delta_trained, 3)
            #     tau_total_str = np.round(tau_hold, 3)
            #     print(f"[STEP {step}] tau_mpc: {tau_mpc_str}, delta_tau: {delta_str}, tau_total: {tau_total_str}")
            
            # Store 6-torque comparison
            torque_comparison_logger.add_sample(
                t=t,
                tau_mpc=tau_mpc_hold,
                tau_true=tau_true,
                delta_trained=delta_trained,
                tau_final=tau_hold  # This is tau_mpc + alpha*clamp(delta_trained)
            )
            
            # Store tracking data
            tracking_logger.add_sample(
                t=t,
                q_ref_dict=q_ref_dict,
                q_act=data.qpos,
                tau=tau_hold,  # Total torque (MPC + NN)
                shoulder_id=shoulder_id
                # upperarm_id=upperarm_id,
                # wrist_id=wrist_id
            )
            
            # Real-time control
            time.sleep(sim_dt * realtime_factor)
    
    # ===============================
    # 10. Stop MPC Thread
    # ===============================
    print("\n⏹️  Stopping MPC+NN thread...")
    mpc_manager.stop()
    print("✅ Simulation finished")
    
    # ===============================
    # 11. Performance Analysis
    # ===============================
    phases = get_trajectory_phases()
    wave_start_time = phases['wait']
    tracking_logger.print_statistics(joint_names, wave_start_time)
    
    # ===============================
    # 12. Visualization
    # ===============================
    print("\n📊 Generating plots...")
    plot_joint_tracking(tracking_logger, joint_names)
    plot_tracking_errors(tracking_logger)
    plot_applied_torques(tracking_logger)
    
    # NEW: 6가지 토크 비교 그래프
    print("\n📊 Generating 6-torque comparison plots...")
    plot_six_torques_comparison(torque_comparison_logger, joint_names)
    plot_all_six_torques(torque_comparison_logger, joint_names)
    plot_residual_accuracy(torque_comparison_logger, joint_names)
    
    print("\n" + "=" * 60)
    print("✅ MPC + Residual NN control completed successfully!")
    print("=" * 60 + "\n")

    result_mpc_only = np.load("dataGet/result_mpc_only.npz", allow_pickle=True)

    evaluator = PerformanceEvaluator(dt=0.005)
    result_mpc_nn  = evaluator.evaluate(tracking_logger,  label="MPC + NN")

    evaluator.compare(result_mpc_only, result_mpc_nn)


if __name__ == "__main__":
    main()
"""
Visualization Tools

시뮬레이션 결과를 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from .trajectory import get_trajectory_phases


def plot_joint_tracking(logger, joint_names):
    """
    Plot joint tracking performance
    
    Args:
        logger: TrackingLogger instance with logged data
        joint_names: List of [shoulder_name, upperarm_name, wrist_name]
    """
    arrays = logger.get_arrays()
    phases = get_trajectory_phases()
    
    fig, axes = plt.subplots(1, 1, figsize=(9, 5))
    
    # Phase lines
    # phase_times = [phases['raise'], phases['wait'], phases['wave']]
    phase_times = [phases['raise'], phases['wait'], phases['hold']]
    
    # # Shoulder
    # axes[0].plot(arrays['time'], arrays['shoulder_ref'], 'r--', 
    #             linewidth=2, label="Reference")
    # axes[0].plot(arrays['time'], arrays['shoulder_act'], 'b-', 
    #             linewidth=1.5, label="Actual (MPC)")
    # for t in phase_times:
    #     axes[0].axvline(t, color='gray', linestyle=':', alpha=0.5)
    # axes[0].set_ylabel("Shoulder Angle [rad]")
    # axes[0].set_title(f"Shoulder Joint ({joint_names[0]}) - MPC Control")
    # axes[0].legend()
    # axes[0].grid(True, alpha=0.3)
        # Shoulder
    
    axes.plot(arrays['time'], arrays['shoulder_ref'], 'r--', 
     linewidth=2, label="Reference")
    axes.plot(arrays['time'], arrays['shoulder_act'], 'b-', 
                linewidth=1.5, label="Actual (MPC)")
    for t in phase_times:
        axes.axvline(t, color='gray', linestyle=':', alpha=0.5)
    axes.set_ylabel("Shoulder Angle [rad]")
    axes.set_title(f"Shoulder Joint ({joint_names[0]}) - MPC Control")
    axes.legend()
    axes.grid(True, alpha=0.3)

    # # Upper Arm
    # axes[1].plot(arrays['time'], arrays['upperarm_ref'], 'r--', 
    #             linewidth=2, label="Reference")
    # axes[1].plot(arrays['time'], arrays['upperarm_act'], 'm-', 
    #             linewidth=1.5, label="Actual (MPC)")
    # for t in phase_times:
    #     axes[1].axvline(t, color='gray', linestyle=':', alpha=0.5)
    # axes[1].set_ylabel("Upper Arm Angle [rad]")
    # axes[1].set_title(f"Upper Arm Joint ({joint_names[1]}) - MPC Control")
    # axes[1].legend()
    # axes[1].grid(True, alpha=0.3)
    
    # # Wrist
    # axes[2].plot(arrays['time'], arrays['wrist_ref'], 'r--', 
    #             linewidth=2, label="Reference")
    # axes[2].plot(arrays['time'], arrays['wrist_act'], 'g-', 
    #             linewidth=1.5, label="Actual (MPC)")
    # for t in phase_times:
    #     axes[2].axvline(t, color='gray', linestyle=':', alpha=0.5)
    # axes[2].set_xlabel("Time [s]")
    # axes[2].set_ylabel("Wrist Angle [rad]")
    # axes[2].set_title(f"Wrist Joint ({joint_names[2]}) - MPC Control")
    # axes[2].legend()
    # axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_tracking_errors(logger):
    """
    Plot tracking errors
    
    Args:
        logger: TrackingLogger instance with logged data
    """
    arrays = logger.get_arrays()
    errors = logger.compute_tracking_errors()
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    
    # axes[0].plot(arrays['time'], errors['shoulder_error'], 'b-')
    # axes[0].set_ylabel("Shoulder Error [rad]")
    # # axes[0].set_title("Tracking Error")
    # # axes[0].grid(True, alpha=0.3)
    axes.plot(arrays['time'], errors['shoulder_error'], 'b-')
    axes.set_ylabel("Shoulder Error [rad]")
    axes.set_title("Tracking Error")
    axes.grid(True, alpha=0.3)
        
    # axes[1].plot(arrays['time'], errors['upperarm_error'], 'm-')
    # axes[1].set_ylabel("Upper Arm Error [rad]")
    # axes[1].grid(True, alpha=0.3)
    
    # axes[2].plot(arrays['time'], errors['wrist_error'], 'g-')
    # axes[2].set_xlabel("Time [s]")
    # axes[2].set_ylabel("Wrist Error [rad]")
    # axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_applied_torques(logger):
    """
    Plot applied torques
    
    Args:
        logger: TrackingLogger instance with logged data
    """
    arrays = logger.get_arrays()
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    
    # axes[0].plot(arrays['time'], arrays['tau_shoulder'], 'b-', linewidth=1.5)
    # axes[0].set_ylabel("Shoulder Torque [Nm]")
    # axes[0].set_title("Applied Torques (MPC)")
    # axes[0].grid(True, alpha=0.3)
    axes.plot(arrays['time'], arrays['tau_shoulder'], 'b-', linewidth=1.5)
    axes.set_ylabel("Shoulder Torque [Nm]")
    axes.set_title("Applied Torques (MPC)")
    axes.grid(True, alpha=0.3)
    
    # axes[1].plot(arrays['time'], arrays['tau_upperarm'], 'm-', linewidth=1.5)
    # axes[1].set_ylabel("Upper Arm Torque [Nm]")
    # axes[1].grid(True, alpha=0.3)
    
    # axes[2].plot(arrays['time'], arrays['tau_wrist'], 'g-', linewidth=1.5)
    # axes[2].set_xlabel("Time [s]")
    # axes[2].set_ylabel("Wrist Torque [Nm]")
    # axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_residual_torque_bar(stats, joint_names):
    """
    Plot residual torque magnitude as bar chart
    
    Args:
        stats: Dictionary from DatasetCollector.get_statistics()
        joint_names: List of [shoulder_name, upperarm_name, wrist_name]
    """
    if stats is None:
        print("No statistics available")
        return
    
    plt.figure(figsize=(6, 4))
    plt.bar(joint_names, stats['mean_abs'])
    plt.ylabel("Mean |Δτ| [Nm]")
    plt.title("Residual Torque Magnitude")
    plt.grid(True)
    plt.show()


def plot_all_results(tracking_logger, dataset_collector, joint_names):
    """
    Plot all visualization results
    
    Args:
        tracking_logger: TrackingLogger instance
        dataset_collector: DatasetCollector instance
        joint_names: List of [shoulder_name, upperarm_name, wrist_name]
    """
    print("\n📊 Generating plots...")
    
    # Plot 1: Joint Tracking
    plot_joint_tracking(tracking_logger, joint_names)
    
    # Plot 2: Tracking Errors
    plot_tracking_errors(tracking_logger)
    
    # Plot 3: Applied Torques
    plot_applied_torques(tracking_logger)
    
    # Plot 4: Residual Torque Bar Chart
    stats = dataset_collector.get_statistics()
    plot_residual_torque_bar(stats, joint_names)
    
    print("✅ All plots displayed")


def plot_six_torques_comparison(torque_logger, joint_names):
    """
    Plot 6 different torques for comparison
    
    6가지 토크:
    1. tau_mpc: MPC가 계산한 토크
    2. tau_true: 실제 필요한 토크
    3. tau_residual: tau_true - tau_mpc (잔여 토크)
    4. delta_trained: 학습된 NN이 예측한 residual
    5. tau_total: tau_mpc + delta_trained
    6. tau_final: tau_mpc + alpha * clamp(delta_trained) (실제 적용)
    
    Args:
        torque_logger: TorqueComparisonLogger instance with logged data
        joint_names: List of [shoulder_name, upperarm_name, wrist_name]
    """
    arrays = torque_logger.get_arrays()
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 12))
    
    joint_labels = ["Shoulder", "Upper Arm", "Wrist"]
    colors = {
        'mpc': '#1f77b4',       # Blue
        'true': '#ff7f0e',      # Orange
        'residual': '#2ca02c',  # Green
        'trained': '#d62728',   # Red
        'total': '#9467bd',     # Purple
        'final': '#8c564b'      # Brown
    }
    
    for joint_idx in range(1):  # 3 joints
        # Plot 1: MPC vs True
        # ax1 = axes[joint_idx, 0]
        axes.plot(arrays['time'], arrays['tau_mpc'][:, joint_idx], 
                label='τ_mpc (MPC)', color=colors['mpc'], linewidth=2)
        axes.plot(arrays['time'], arrays['tau_true'][:, joint_idx], 
             label='τ_true (Required)', color=colors['true'], linewidth=2, linestyle='--')
        axes.set_ylabel(f'{joint_labels[joint_idx]} Torque [Nm]', fontsize=11)
        axes.set_title(f'{joint_labels[joint_idx]} ({joint_names[joint_idx]}): MPC vs True', fontsize=12, fontweight='bold')
        axes.legend(loc='upper right', fontsize=9)
        axes.grid(True, alpha=0.3)
        
        # # Plot 2: Residual Comparison
        # ax2 = axes[joint_idx, 1]
        # ax2.plot(arrays['time'], arrays['tau_residual'][:, joint_idx], 
        #         label='Δτ_true (true - mpc)', color=colors['residual'], linewidth=2)
        # ax2.plot(arrays['time'], arrays['delta_trained'][:, joint_idx], 
        #         label='Δτ_NN (trained)', color=colors['trained'], linewidth=2, linestyle='--')
        # ax2.set_ylabel(f'Residual Torque [Nm]', fontsize=11)
        # ax2.set_title(f'{joint_labels[joint_idx]}: Residual Comparison', fontsize=12, fontweight='bold')
        # ax2.legend(loc='upper right', fontsize=9)
        # ax2.grid(True, alpha=0.3)
        
        # # Only add xlabel to bottom plots
        # if joint_idx == 2:
        #     ax1.set_xlabel('Time [s]', fontsize=11)
        #     ax2.set_xlabel('Time [s]', fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_all_six_torques(torque_logger, joint_names):
    """
    Plot all 6 torques in a single figure for each joint
    
    Args:
        torque_logger: TorqueComparisonLogger instance
        joint_names: List of [shoulder_name, upperarm_name, wrist_name]
    """
    arrays = torque_logger.get_arrays()
    
    fig, axes = plt.subplots(1, 1, figsize=(14, 12))
    
    joint_labels = ["Shoulder", "Upper Arm", "Wrist"]
    
    for joint_idx in range(1):
        # ax = axes[joint_idx]
        
        # Plot all 6 torques
        axes.plot(arrays['time'], arrays['tau_mpc'][:, joint_idx], 
               label='1. τ_mpc (MPC)', linewidth=2, alpha=0.8)
        axes.plot(arrays['time'], arrays['tau_true'][:, joint_idx], 
               label='2. τ_true (Required)', linewidth=2, linestyle='--', alpha=0.8)
        axes.plot(arrays['time'], arrays['tau_residual'][:, joint_idx], 
               label='3. Δτ_true (true - mpc)', linewidth=1.5, alpha=0.7)
        axes.plot(arrays['time'], arrays['delta_trained'][:, joint_idx], 
               label='4. Δτ_NN (trained)', linewidth=1.5, linestyle=':', alpha=0.7)
        axes.plot(arrays['time'], arrays['tau_total'][:, joint_idx], 
               label='5. τ_total (mpc + NN)', linewidth=2, alpha=0.8)
        axes.plot(arrays['time'], arrays['tau_final'][:, joint_idx], 
               label='6. τ_final (applied)', linewidth=2.5, alpha=0.9)
        
        axes.set_ylabel(f'{joint_labels[joint_idx]} Torque [Nm]', fontsize=11)
        axes.set_title(f'{joint_labels[joint_idx]} ({joint_names[joint_idx]}): All 6 Torques', 
                    fontsize=12, fontweight='bold')
        axes.legend(loc='best', fontsize=9, ncol=2)
        axes.grid(True, alpha=0.3)
        
        # if joint_idx == 2:
        #     ax.set_xlabel('Time [s]', fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_residual_accuracy(torque_logger, joint_names):
    """
    Plot residual prediction accuracy (NN vs True)
    
    Args:
        torque_logger: TorqueComparisonLogger instance
        joint_names: List of joint names
    """
    arrays = torque_logger.get_arrays()
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    
    joint_labels = ["Shoulder", "Upper Arm", "Wrist"]
    
    for joint_idx in range(1):
        # ax = axes[joint_idx]
        
        true_res = arrays['tau_residual'][:, joint_idx]
        nn_res = arrays['delta_trained'][:, joint_idx]
        error = true_res - nn_res
        
        # Plot true vs predicted
        axes.plot(arrays['time'], true_res, label='Δτ_true', linewidth=2, alpha=0.8)
        axes.plot(arrays['time'], nn_res, label='Δτ_NN', linewidth=2, linestyle='--', alpha=0.8)
        axes.fill_between(arrays['time'], true_res, nn_res, alpha=0.2, label='Error')
        
        # Calculate metrics
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        
        axes.set_ylabel(f'{joint_labels[joint_idx]} Residual [Nm]', fontsize=11)
        axes.set_title(f'{joint_labels[joint_idx]} ({joint_names[joint_idx]}): '
                    f'RMSE={rmse:.3f} Nm, MAE={mae:.3f} Nm', 
                    fontsize=12, fontweight='bold')
        axes.legend(loc='upper right', fontsize=10)
        axes.grid(True, alpha=0.3)
        
        # if joint_idx == 2:
        #     ax.set_xlabel('Time [s]', fontsize=11)
    
    plt.tight_layout()
    plt.show()


def print_residual_analysis(stats, joint_names):
    """
    Print residual torque analysis
    
    Args:
        stats: Dictionary from DatasetCollector.get_statistics()
        joint_names: List of [shoulder_name, upperarm_name, wrist_name]
    """
    if stats is None:
        print("No statistics available")
        return
    
    print("\n[Δτ Analysis]")
    print(f"Joint order: {', '.join(joint_names)}")
    print(f"Mean |Δτ|: {stats['mean_abs']}")
    print(f"Max  |Δτ|: {stats['max_abs']}")
    print(f"Std  |Δτ|: {stats['std_abs']}")
    print(f"Mean |Δτ| / Mean |τ_mpc|: {stats['ratio']}")
    print(f"Total samples: {stats['n_samples']}")
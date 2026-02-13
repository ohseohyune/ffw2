"""
SLSQP vs iLQR 성능 비교 스크립트

두 최적화 방법의 실제 성능을 측정하고 비교합니다.
"""

import numpy as np
import mujoco
import time
from dataclasses import dataclass
from typing import List, Dict

# Import both controllers
from mpc_controller import TorqueMPC  # SLSQP version
from mpc_controller_ilqr import create_ilqr_mpc  # iLQR version

from config import (
    PathConfig, SimulationConfig, MPCConfig, 
    CostWeights, TorqueLimits
)


@dataclass
class BenchmarkResult:
    """Benchmark result container"""
    name: str
    solve_times: List[float]
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: List[int]
    mean_iter: float


class MPCBenchmark:
    """
    MPC 컨트롤러 벤치마크
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to MuJoCo XML file
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Setup robot (simplified for benchmark)
        from robot_setup import setup_robot
        robot = setup_robot(self.model)
        self.joint_ids = robot.get_controlled_joint_ids()
        
        # Simulation parameters
        self.sim_dt = SimulationConfig.SIM_DT
        self.horizon = MPCConfig.HORIZON
        
        print(f"\n📊 Benchmark Setup:")
        print(f"   Model: {model_path}")
        print(f"   Controlled joints: {len(self.joint_ids)}")
        print(f"   Horizon: {self.horizon}")
        print(f"   dt: {self.sim_dt*1000:.1f} ms")
    
    def benchmark_slsqp(self, n_tests: int = 50) -> BenchmarkResult:
        """
        Benchmark SLSQP controller
        
        Args:
            n_tests: Number of test iterations
        
        Returns:
            BenchmarkResult
        """
        print(f"\n🔵 Benchmarking SLSQP...")
        
        # Create controller
        controller = TorqueMPC(
            model=self.model,
            joint_ids=self.joint_ids,
            horizon=self.horizon,
            dt=self.sim_dt
        )
        
        # Test data
        q_full = self.data.qpos.copy()
        qdot_full = self.data.qvel.copy()
        q_ref = np.zeros(len(self.joint_ids))
        q_ref_prev = np.zeros(len(self.joint_ids))
        
        solve_times = []
        iterations = []
        
        # Warm-up run
        controller.compute_control_from_state(
            q_full, qdot_full, q_ref, q_ref_prev
        )
        
        # Benchmark runs
        for i in range(n_tests):
            t0 = time.perf_counter()
            _, _, nit = controller.compute_control_from_state(
                q_full, qdot_full, q_ref, q_ref_prev
            )
            solve_time = time.perf_counter() - t0
            
            solve_times.append(solve_time * 1000)  # Convert to ms
            iterations.append(nit)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{n_tests}")
        
        solve_times = np.array(solve_times)
        iterations = np.array(iterations)
        
        result = BenchmarkResult(
            name="SLSQP",
            solve_times=solve_times.tolist(),
            mean_time=np.mean(solve_times),
            std_time=np.std(solve_times),
            min_time=np.min(solve_times),
            max_time=np.max(solve_times),
            iterations=iterations.tolist(),
            mean_iter=np.mean(iterations)
        )
        
        print(f"✅ SLSQP benchmark complete")
        
        return result
    
    def benchmark_ilqr(self, n_tests: int = 50) -> BenchmarkResult:
        """
        Benchmark iLQR controller
        
        Args:
            n_tests: Number of test iterations
        
        Returns:
            BenchmarkResult
        """
        print(f"\n🟢 Benchmarking iLQR...")
        
        # Prepare config
        config = {
            'Q_pos': CostWeights.Q_POS,
            'Q_vel': CostWeights.Q_VEL,
            'Q_vel_ref': CostWeights.Q_VEL_REF,
            'R_tau': CostWeights.R_TAU,
            'Q_terminal': CostWeights.Q_TERMINAL,
            'Q_vel_terminal': CostWeights.Q_VEL_TERMINAL,
            'tau_max': TorqueLimits.TAU_MAX,
            'tau_min': TorqueLimits.TAU_MIN,
        }
        
        # Create controller
        controller = create_ilqr_mpc(
            model=self.model,
            joint_ids=self.joint_ids,
            horizon=self.horizon,
            dt=self.sim_dt,
            config=config
        )
        
        # Test data
        q_full = self.data.qpos.copy()
        qdot_full = self.data.qvel.copy()
        q_ref = np.zeros(len(self.joint_ids))
        q_ref_prev = np.zeros(len(self.joint_ids))
        
        solve_times = []
        iterations = []
        
        # Warm-up run (for Numba compilation)
        print("   ⏳ Warm-up (Numba compilation)...")
        controller.compute_control_from_state(
            q_full, qdot_full, q_ref, q_ref_prev
        )
        print("   ✅ Compilation complete")
        
        # Benchmark runs
        for i in range(n_tests):
            t0 = time.perf_counter()
            _, _, nit = controller.compute_control_from_state(
                q_full, qdot_full, q_ref, q_ref_prev
            )
            solve_time = time.perf_counter() - t0
            
            solve_times.append(solve_time * 1000)  # Convert to ms
            iterations.append(nit)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{n_tests}")
        
        solve_times = np.array(solve_times)
        iterations = np.array(iterations)
        
        result = BenchmarkResult(
            name="iLQR",
            solve_times=solve_times.tolist(),
            mean_time=np.mean(solve_times),
            std_time=np.std(solve_times),
            min_time=np.min(solve_times),
            max_time=np.max(solve_times),
            iterations=iterations.tolist(),
            mean_iter=np.mean(iterations)
        )
        
        print(f"✅ iLQR benchmark complete")
        
        return result
    
    def print_comparison(self, slsqp_result: BenchmarkResult, 
                        ilqr_result: BenchmarkResult):
        """
        Print comparison results
        
        Args:
            slsqp_result: SLSQP benchmark result
            ilqr_result: iLQR benchmark result
        """
        print("\n" + "=" * 70)
        print("📊 BENCHMARK RESULTS")
        print("=" * 70)
        
        print(f"\n{'Metric':<30} {'SLSQP':<20} {'iLQR':<20}")
        print("-" * 70)
        
        print(f"{'Mean solve time (ms)':<30} "
              f"{slsqp_result.mean_time:>18.2f}  "
              f"{ilqr_result.mean_time:>18.2f}")
        
        print(f"{'Std deviation (ms)':<30} "
              f"{slsqp_result.std_time:>18.2f}  "
              f"{ilqr_result.std_time:>18.2f}")
        
        print(f"{'Min solve time (ms)':<30} "
              f"{slsqp_result.min_time:>18.2f}  "
              f"{ilqr_result.min_time:>18.2f}")
        
        print(f"{'Max solve time (ms)':<30} "
              f"{slsqp_result.max_time:>18.2f}  "
              f"{ilqr_result.max_time:>18.2f}")
        
        print(f"{'Mean iterations':<30} "
              f"{slsqp_result.mean_iter:>18.1f}  "
              f"{ilqr_result.mean_iter:>18.1f}")
        
        print("-" * 70)
        
        # Speedup calculation
        speedup = slsqp_result.mean_time / ilqr_result.mean_time
        
        print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster with iLQR!")
        
        # Feasibility analysis
        control_freq_slsqp = 1000 / slsqp_result.mean_time  # Hz
        control_freq_ilqr = 1000 / ilqr_result.mean_time  # Hz
        
        print(f"\n📈 Maximum Control Frequency:")
        print(f"   SLSQP: {control_freq_slsqp:.1f} Hz")
        print(f"   iLQR:  {control_freq_ilqr:.1f} Hz")
        
        if ilqr_result.mean_time < 5.0:
            print(f"\n✅ iLQR is suitable for 200 Hz control!")
        elif ilqr_result.mean_time < 10.0:
            print(f"\n✅ iLQR is suitable for 100 Hz control!")
        else:
            print(f"\n⚠️  iLQR may need optimization for high-frequency control")
        
        print("\n" + "=" * 70)
    
    def save_results(self, slsqp_result: BenchmarkResult, 
                    ilqr_result: BenchmarkResult, 
                    filename: str = "benchmark_results.npz"):
        """
        Save benchmark results
        
        Args:
            slsqp_result: SLSQP benchmark result
            ilqr_result: iLQR benchmark result
            filename: Output filename
        """
        np.savez(
            filename,
            slsqp_solve_times=slsqp_result.solve_times,
            slsqp_iterations=slsqp_result.iterations,
            slsqp_mean_time=slsqp_result.mean_time,
            slsqp_std_time=slsqp_result.std_time,
            ilqr_solve_times=ilqr_result.solve_times,
            ilqr_iterations=ilqr_result.iterations,
            ilqr_mean_time=ilqr_result.mean_time,
            ilqr_std_time=ilqr_result.std_time,
            speedup=slsqp_result.mean_time / ilqr_result.mean_time
        )
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Run benchmark comparison"""
    
    print("\n" + "=" * 70)
    print("🏁 SLSQP vs iLQR MPC Benchmark")
    print("=" * 70)
    
    # Get model path
    paths = PathConfig.get_paths()
    xml_path = paths['xml_path']
    
    # Create benchmark
    benchmark = MPCBenchmark(xml_path)
    
    # Run benchmarks
    n_tests = 100
    print(f"\nRunning {n_tests} iterations for each controller...")
    
    slsqp_result = benchmark.benchmark_slsqp(n_tests)
    ilqr_result = benchmark.benchmark_ilqr(n_tests)
    
    # Print comparison
    benchmark.print_comparison(slsqp_result, ilqr_result)
    
    # Save results
    benchmark.save_results(slsqp_result, ilqr_result)
    
    # Plot results (optional)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Solve time distribution
        axes[0, 0].hist(slsqp_result.solve_times, bins=30, alpha=0.7, 
                       label='SLSQP', color='blue')
        axes[0, 0].hist(ilqr_result.solve_times, bins=30, alpha=0.7, 
                       label='iLQR', color='green')
        axes[0, 0].set_xlabel('Solve Time (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Solve Time Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Time series
        axes[0, 1].plot(slsqp_result.solve_times, 'b-', alpha=0.5, 
                       label='SLSQP', linewidth=0.5)
        axes[0, 1].plot(ilqr_result.solve_times, 'g-', alpha=0.5, 
                       label='iLQR', linewidth=0.5)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Solve Time (ms)')
        axes[0, 1].set_title('Solve Time vs Iteration')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Box plot
        axes[1, 0].boxplot([slsqp_result.solve_times, ilqr_result.solve_times],
                          labels=['SLSQP', 'iLQR'])
        axes[1, 0].set_ylabel('Solve Time (ms)')
        axes[1, 0].set_title('Solve Time Box Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative distribution
        axes[1, 1].hist(slsqp_result.solve_times, bins=50, cumulative=True,
                       density=True, alpha=0.7, label='SLSQP', color='blue')
        axes[1, 1].hist(ilqr_result.solve_times, bins=50, cumulative=True,
                       density=True, alpha=0.7, label='iLQR', color='green')
        axes[1, 1].set_xlabel('Solve Time (ms)')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmark_comparison.png', dpi=150)
        print("\n📊 Plots saved to: benchmark_comparison.png")
        plt.show()
        
    except ImportError:
        print("\n⚠️  Matplotlib not found, skipping plots")


if __name__ == "__main__":
    # Check dependencies
    try:
        import ilqr
    except ImportError:
        print("\n" + "="*70)
        print("❌ ERROR: iLQR library not found!")
        print("="*70)
        print("\nPlease install it:")
        print("  pip install git+https://github.com/Bharath2/iLQR.git")
        print("="*70 + "\n")
        exit(1)
    
    main()
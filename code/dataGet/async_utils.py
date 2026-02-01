"""
Asynchronous MPC Utilities

Thread-safe buffers and worker for async MPC execution
"""

import numpy as np
import threading
import time
from dataclasses import dataclass


@dataclass
class MPCInput:
    """
    MPC 입력 데이터 구조
    
    Attributes:
        q: Full joint positions
        qdot: Full joint velocities
        q_ref: Reference positions for controlled joints
        q_ref_prev: Previous reference positions (for velocity estimation)
        stamp: Timestamp
    """
    q: np.ndarray
    qdot: np.ndarray
    q_ref: np.ndarray
    q_ref_prev: np.ndarray
    stamp: float


class SharedTorqueBuffer:
    """
    Thread-safe buffer for MPC output torques
    
    Latest torque output from MPC.
    Simulation reads fast; MPC writes when done.
    """
    
    def __init__(self, nq: int):
        """
        Args:
            nq: Number of controlled joints
        """
        self._lock = threading.Lock()
        self._tau_total = np.zeros(nq)
        self._tau_mpc = np.zeros(nq)
        self._stamp = 0.0
        self._has_value = False

    def write(self, tau_total: np.ndarray, tau_mpc: np.ndarray, stamp: float):
        """
        Write new torque values (called by MPC thread)
        
        Args:
            tau_total: Total torque (MPC + bias)
            tau_mpc: MPC torque only
            stamp: Timestamp
        """
        with self._lock:
            self._tau_total[:] = tau_total
            self._tau_mpc[:] = tau_mpc
            self._stamp = stamp
            self._has_value = True

    def read_latest(self):
        """
        Read latest torque values (called by simulation thread)
        
        Returns:
            ok: True if valid data exists
            tau_total: Total torque
            tau_mpc: MPC torque
            stamp: Timestamp
        """
        with self._lock:
            tau_total = self._tau_total.copy()
            tau_mpc = self._tau_mpc.copy()
            stamp = self._stamp
            ok = self._has_value
        return ok, tau_total, tau_mpc, stamp


class SharedMPCInput:
    """
    Thread-safe buffer for MPC input
    
    Latest MPC input (state + ref). Overwritten by simulation periodically.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._inp: MPCInput | None = None

    def write(self, inp: MPCInput):
        """
        Write new input data (called by simulation thread)
        
        Args:
            inp: MPCInput data
        """
        with self._lock:
            self._inp = inp

    def read_latest(self):
        """
        Read latest input data (called by MPC thread)
        
        Returns:
            MPCInput or None if no data available
        """
        with self._lock:
            if self._inp is None:
                return None
            return MPCInput(
                q=self._inp.q.copy(),
                qdot=self._inp.qdot.copy(),
                q_ref=self._inp.q_ref.copy(),
                q_ref_prev=self._inp.q_ref_prev.copy(),
                stamp=self._inp.stamp
            )


def mpc_worker(model, controller,
               shared_inp: SharedMPCInput,
               shared_tau: SharedTorqueBuffer,
               stop_event: threading.Event,
               mpc_rate_hz: float):
    """
    MPC Worker Thread
    
    Runs MPC optimization at specified rate, independent of simulation
    
    Args:
        model: MuJoCo model
        controller: TorqueMPC controller instance
        shared_inp: Shared input buffer
        shared_tau: Shared torque buffer
        stop_event: Thread stop signal
        mpc_rate_hz: MPC execution frequency [Hz]
    """
    period = 1.0 / max(mpc_rate_hz, 1e-6)
    next_time = time.time()

    while not stop_event.is_set():
        now = time.time()
        
        # Wait until next scheduled time
        if now < next_time:
            time.sleep(min(0.001, next_time - now))
            continue
        
        next_time += period

        # Read latest input
        inp = shared_inp.read_latest()
        if inp is None:
            continue

        # Run MPC optimization
        t0 = time.time()
        try:
            tau_total, tau_mpc = controller.compute_control_from_state(
                inp.q, inp.qdot, inp.q_ref, inp.q_ref_prev
            )
            
            # Write result
            shared_tau.write(tau_total, tau_mpc, stamp=time.time())
            
            solve_time = (time.time() - t0) * 1000
            print(f"[MPC] solve: {solve_time:.1f}ms")
            
        except Exception as e:
            print(f"[MPC] solve failed: {e}")


class MPCAsyncManager:
    """
    Manager for asynchronous MPC execution
    
    Handles thread creation, communication buffers, and lifecycle
    """
    
    def __init__(self, model, controller, mpc_rate_hz):
        """
        Args:
            model: MuJoCo model
            controller: TorqueMPC controller instance
            mpc_rate_hz: MPC execution frequency [Hz]
        """
        self.model = model
        self.controller = controller
        self.mpc_rate_hz = mpc_rate_hz
        
        # Create shared buffers
        self.shared_inp = SharedMPCInput()
        self.shared_tau = SharedTorqueBuffer(nq=controller.nq)
        self.stop_event = threading.Event()
        
        # Thread reference
        self.mpc_thread = None

    def start(self):
        """Start MPC worker thread"""
        if self.mpc_thread is not None and self.mpc_thread.is_alive():
            print("[MPCAsyncManager] Thread already running")
            return
        
        self.stop_event.clear()
        self.mpc_thread = threading.Thread(
            target=mpc_worker,
            args=(
                self.model,
                self.controller,
                self.shared_inp,
                self.shared_tau,
                self.stop_event,
                self.mpc_rate_hz
            ),
            daemon=True
        )
        self.mpc_thread.start()
        print(f"[MPCAsyncManager] Started MPC thread at {self.mpc_rate_hz} Hz")

    def stop(self, timeout=1.0):
        """
        Stop MPC worker thread
        
        Args:
            timeout: Maximum time to wait for thread to stop [s]
        """
        if self.mpc_thread is None:
            return
        
        self.stop_event.set()
        self.mpc_thread.join(timeout=timeout)
        print("[MPCAsyncManager] Stopped MPC thread")

    def push_input(self, q, qdot, q_ref, q_ref_prev, stamp):
        """
        Push new input to MPC thread
        
        Args:
            q: Full joint positions
            qdot: Full joint velocities
            q_ref: Reference positions for controlled joints
            q_ref_prev: Previous reference positions for controlled joints
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
        """
        Read latest torque from MPC thread
        
        Returns:
            ok: True if valid data exists
            tau_total: Total torque
            tau_mpc: MPC torque
            stamp: Timestamp
        """
        return self.shared_tau.read_latest()
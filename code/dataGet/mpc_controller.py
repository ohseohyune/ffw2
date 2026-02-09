"""
Torque-based Model Predictive Control

MuJoCo 로봇을 위한 MPC 컨트롤러입니다.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize
from .config import MPCConfig, CostWeights, TorqueLimits

            
class TorqueMPC:
    """
    Model Predictive Control for torque-based robot control
    
    Dedicated MPC data로 thread-safe하게 동작합니다.
    RK4 integration과 horizon-level dynamics caching 사용.
    """
    
    def __init__(self, model, joint_ids, horizon=None, dt=None):
        """
        Args:
            model: MuJoCo model
            joint_ids: List of joint indices to control (qpos indices)
            horizon: MPC prediction horizon (uses config if None)
            dt: Time step for prediction (uses config if None)
        """
        self.model = model
        self.joint_ids = joint_ids
        self.nq = len(joint_ids)
        
        # Use config values if not provided
        if horizon is None:
            horizon = MPCConfig.HORIZON
        if dt is None:
            from config import SimulationConfig
            dt = SimulationConfig.SIM_DT
            
        self.horizon = horizon
        self.dt = dt

        # Cost weights
        self.Q_pos = CostWeights.Q_POS.copy()
        self.Q_vel = CostWeights.Q_VEL.copy()
        self.Q_vel_ref = CostWeights.Q_VEL_REF.copy()
        self.R_tau = CostWeights.R_TAU.copy()
        self.Q_terminal = CostWeights.Q_TERMINAL.copy()
        self.Q_vel_terminal = CostWeights.Q_VEL_TERMINAL.copy()

        # Torque limits
        self.tau_max = TorqueLimits.TAU_MAX * np.ones(self.nq)
        self.tau_min = TorqueLimits.TAU_MIN * np.ones(self.nq)

        # Dedicated MPC data (thread-only)
        self.data_mpc = mujoco.MjData(model)

        # Full DOF mapping
        self.full_nv = model.nv

        # Cached dynamics (computed once per horizon)
        self.M_inv_cache = None
        self.bias_cache = None

    def _cache_dynamics_from_state(self, q_full, qdot_full):
        """
        Cache dynamics using mj_forward for consistency
        Extracts submatrix for controlled joints
        Called ONCE per optimization horizon
        
        Args:
            q_full: Full joint positions
            qdot_full: Full joint velocities
        """
        # Set full state into data_mpc
        self.data_mpc.qpos[:] = q_full
        self.data_mpc.qvel[:] = qdot_full

        # Make all derived quantities consistent
        mujoco.mj_forward(self.model, self.data_mpc)

        # Build full mass matrix
        M_full = np.zeros((self.full_nv, self.full_nv))
        mujoco.mj_fullM(self.model, M_full, self.data_mpc.qM)

        # Extract submatrix for controlled joints
        M_sub = M_full[np.ix_(self.joint_ids, self.joint_ids)]
        bias_full = self.data_mpc.qfrc_bias.copy()
        bias_sub = bias_full[self.joint_ids]

        # Cache inverse and bias
        self.M_inv_cache = np.linalg.inv(M_sub)
        self.bias_cache = bias_sub

    def _compute_acceleration(self, tau):
        """
        Compute acceleration using cached dynamics
        
        Args:
            tau: Applied torques
        
        Returns:
            qddot: Joint accelerations
        """
        
        # qddot_real = self.data_mpc.qacc[self.joint_ids].copy()

        # return qddot_real
        return self.M_inv_cache @ (tau - self.bias_cache)

    def _rk4_step(self, q, qdot, tau):
        """
        Runge-Kutta 4th order integration step
        
        Args:
            q: Joint positions
            qdot: Joint velocities
            tau: Applied torques
        
        Returns:
            q_next: Next joint positions
            qdot_next: Next joint velocities
        """
        # k1
        k1_qdot = qdot
        k1_qddot = self._compute_acceleration(tau)
        
        # k2
        q2 = q + 0.5 * self.dt * k1_qdot
        qdot2 = qdot + 0.5 * self.dt * k1_qddot
        k2_qdot = qdot2
        k2_qddot = self._compute_acceleration(tau)  # Using same dynamics
        
        # k3
        q3 = q + 0.5 * self.dt * k2_qdot
        qdot3 = qdot + 0.5 * self.dt * k2_qddot
        k3_qdot = qdot3
        k3_qddot = self._compute_acceleration(tau)  # Using same dynamics
        
        # k4
        q4 = q + self.dt * k3_qdot
        qdot4 = qdot + self.dt * k3_qddot
        k4_qdot = qdot4
        k4_qddot = self._compute_acceleration(tau)  # Using same dynamics
        
        # Combine
        q_next = q + (self.dt / 6.0) * (k1_qdot + 2*k2_qdot + 2*k3_qdot + k4_qdot)
        qdot_next = qdot + (self.dt / 6.0) * (k1_qddot + 2*k2_qddot + 2*k3_qddot + k4_qddot)
        
        return q_next, qdot_next

    def _compute_reference_velocity(self, q_ref, q_prev_ref):
        """
        Compute reference velocity from position references
        Simple finite difference approximation
        
        Args:
            q_ref: Current reference position
            q_prev_ref: Previous reference position
        
        Returns:
            qdot_ref: Reference velocity
        """
        return (q_ref - q_prev_ref) / self.dt

    def _compute_cost(self, tau_seq, q0, qdot0, q_ref, qdot_ref):
        """
        Compute total cost for given torque sequence
        
        Args:
            tau_seq: Flattened torque sequence [horizon * nq]
            q0: Initial joint positions
            qdot0: Initial joint velocities
            q_ref: Reference joint positions
            qdot_ref: Reference joint velocities
        
        Returns:
            cost: Total cost value
        """
        q, qdot = q0.copy(), qdot0.copy()
        cost = 0.0
        tau_seq = tau_seq.reshape(self.horizon, self.nq)

        # Running cost over horizon
        for k in range(self.horizon):
            # Position tracking cost
            q_error = q - q_ref
            cost += q_error.T @ self.Q_pos @ q_error
            
            # Velocity reference tracking cost
            qdot_error = qdot - qdot_ref
            cost += qdot_error.T @ self.Q_vel_ref @ qdot_error
            
            # Velocity damping cost (penalize high velocities)
            cost += qdot.T @ self.Q_vel @ qdot
            
            # Control effort cost
            cost += tau_seq[k].T @ self.R_tau @ tau_seq[k]
            
            # Predict next state using RK4
            q, qdot = self._rk4_step(q, qdot, tau_seq[k])

        # Terminal cost
        q_error_final = q - q_ref
        cost += q_error_final.T @ self.Q_terminal @ q_error_final
        
        qdot_error_final = qdot - qdot_ref
        cost += qdot_error_final.T @ self.Q_vel_terminal @ qdot_error_final
        
        return cost

    def compute_control_from_state(self, q_full, qdot_full, q_ref_sub, q_prev_ref_sub=None):
        """
        Compute control from given full state
        
        Args:
            q_full: Full joint positions
            qdot_full: Full joint velocities
            q_ref_sub: Reference positions for controlled joints
            q_prev_ref_sub: Previous reference positions (for velocity estimation)
                           If None, assumes zero reference velocity
        
        Returns:
            tau_total: Total torque (MPC + bias compensation)
            tau_opt: Optimized MPC torque (without bias)
        """
        # Extract controlled joints
        q0 = q_full[self.joint_ids]
        qdot0 = qdot_full[self.joint_ids]

        # Cache dynamics ONCE for this horizon
        self._cache_dynamics_from_state(q_full, qdot_full)

        # Estimate reference velocity
        if q_prev_ref_sub is None:
            qdot_ref = np.zeros(self.nq)
        else:
            qdot_ref = self._compute_reference_velocity(q_ref_sub, q_prev_ref_sub)

        # Initial guess (zero torques)
        tau_init = np.zeros(self.horizon * self.nq)
        
        # Set up bounds
        bounds = [(self.tau_min[i % self.nq], self.tau_max[i % self.nq])
                  for i in range(self.horizon * self.nq)]

        # Optimize
        result = minimize(
            lambda tau: self._compute_cost(tau, q0, qdot0, q_ref_sub, qdot_ref),
            tau_init,
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": MPCConfig.MAX_ITER,
                "ftol": MPCConfig.FTOL
            }
        )

        # Extract first control input and clip
        tau_opt = np.clip(result.x[:self.nq], self.tau_min, self.tau_max)
        
        # tau_opt already accounts for system dynamics in the optimization
        # Return as-is for MuJoCo (no double bias compensation)
        return tau_opt.copy(), tau_opt.copy()

    def update_cost_weights(self, Q_pos=None, Q_vel_ref=None, R_tau=None, Q_terminal=None):
        """
        Update cost function weights
        
        Args:
            Q_pos: Position tracking weight matrix
            Q_vel_ref: Velocity reference tracking weight matrix
            R_tau: Control effort weight matrix
            Q_terminal: Terminal cost weight matrix
        """
        if Q_pos is not None:
            self.Q_pos = Q_pos.copy()
        if Q_vel_ref is not None:
            self.Q_vel_ref = Q_vel_ref.copy()
        if R_tau is not None:
            self.R_tau = R_tau.copy()
        if Q_terminal is not None:
            self.Q_terminal = Q_terminal.copy()

    def get_config(self):
        """Return controller configuration"""
        return {
            'horizon': self.horizon,
            'dt': self.dt,
            'nq': self.nq,
            'joint_ids': self.joint_ids,
            'tau_max': self.tau_max.tolist(),
            'tau_min': self.tau_min.tolist()
        }
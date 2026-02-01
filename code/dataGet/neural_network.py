"""
Residual Torque Neural Network

MPC 토크를 보정하기 위한 신경망 모델입니다.
"""

import torch
import torch.nn as nn
from .config import ResidualNNConfig


class ResidualTorqueNN(nn.Module):
    """
    Residual Torque Neural Network
    
    9차원 입력(상태 + baseline 토크)을 받아서, 
    3개 관절에 대한 residual torque(보정 토크) Δτ를 출력합니다.
    
    입력: [q(3), qdot(3), tau_mpc(3)] = 9차원
    출력: [delta_tau(3)] = 3차원
    """
    
    def __init__(self, delta_tau_max=None):
        """
        Args:
            delta_tau_max: Maximum residual torque magnitude [Nm]
                          If None, uses value from config
        """
        super().__init__()
        
        if delta_tau_max is None:
            delta_tau_max = ResidualNNConfig.DELTA_TAU_MAX
        
        self.delta_tau_max = delta_tau_max
        
        cfg = ResidualNNConfig
        
        self.net = nn.Sequential(
            nn.Linear(cfg.INPUT_DIM, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, cfg.OUTPUT_DIM),
            nn.Tanh()  # Output in [-1, 1], 출력 범위를 강제로 제한
        )

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 9) or (9,)
               [q(3), qdot(3), tau_mpc(3)]
        
        Returns:
            delta_tau: Residual torque tensor of shape (batch_size, 3) or (3,)
                      Range: [-delta_tau_max, delta_tau_max]
        """
        return self.delta_tau_max * self.net(x)
    
    def get_config(self):
        """Return model configuration"""
        return {
            'delta_tau_max': self.delta_tau_max,
            'input_dim': ResidualNNConfig.INPUT_DIM,
            'hidden_dim': ResidualNNConfig.HIDDEN_DIM,
            'output_dim': ResidualNNConfig.OUTPUT_DIM
        }


def create_residual_nn(delta_tau_max=None):
    """
    Factory function to create ResidualTorqueNN
    
    Args:
        delta_tau_max: Maximum residual torque magnitude [Nm]
    
    Returns:
        ResidualTorqueNN instance
    """
    return ResidualTorqueNN(delta_tau_max=delta_tau_max)
"""
Robot Setup

로봇의 관절과 모터를 설정하고 ID를 가져옵니다.
"""

import mujoco
from .config import RobotConfig


class RobotInterface:
    """
    Robot Interface for MuJoCo
    
    관절과 모터 정보를 관리합니다.
    """
    
    def __init__(self, model):
        """
        Args:
            model: MuJoCo model
        """
        self.model = model
        
        # Get joint names from config
        cfg = RobotConfig
        self.shoulder_joint_name = cfg.SHOULDER_JOINT_NAME
        # self.upperarm_joint_name = cfg.UPPERARM_JOINT_NAME
        # self.wrist_joint_name = cfg.WRIST_JOINT_NAME
        self.motor_names = cfg.MOTOR_NAMES.copy()
        
        # Get joint IDs
        self.shoulder_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, self.shoulder_joint_name
        )
        # self.upperarm_joint_id = mujoco.mj_name2id(
        #     model, mujoco.mjtObj.mjOBJ_JOINT, self.upperarm_joint_name
        # )
        # self.wrist_joint_id = mujoco.mj_name2id(
        #     model, mujoco.mjtObj.mjOBJ_JOINT, self.wrist_joint_name
        # )
        
        # Get qpos indices
        self.shoulder_qpos_id = model.jnt_qposadr[self.shoulder_joint_id]
        # self.upperarm_qpos_id = model.jnt_qposadr[self.upperarm_joint_id]
        # self.wrist_qpos_id = model.jnt_qposadr[self.wrist_joint_id]
        
        # Controlled joint IDs for MPC
        self.controlled_joint_ids = [
            self.shoulder_qpos_id,
            # self.upperarm_qpos_id,
            # self.wrist_qpos_id
        ]
        
        # Get actuator IDs
        self.motor_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.motor_names
        ]

    def get_joint_names(self):
        """
        Get joint names as list
        
        Returns:
            list: [shoulder_name, upperarm_name, wrist_name]
        """
        return [
            self.shoulder_joint_name,
            # self.upperarm_joint_name,
            # self.wrist_joint_name
        ]

    def get_qpos_ids(self):
        """
        Get qpos indices
        
        Returns:
            tuple: (shoulder_id, upperarm_id, wrist_id)
        """
        return (
            self.shoulder_qpos_id,
            # self.upperarm_qpos_id,
            # self.wrist_qpos_id
        )

    def get_controlled_joint_ids(self):
        """
        Get controlled joint IDs for MPC
        
        Returns:
            list: [shoulder_qpos_id, upperarm_qpos_id, wrist_qpos_id]
        """
        return self.controlled_joint_ids.copy()

    def get_motor_ids(self):
        """
        Get motor actuator IDs
        
        Returns:
            list: Motor IDs
        """
        return self.motor_ids.copy()

    def apply_torques(self, data, tau):
        """
        Apply torques to motors
        
        Args:
            data: MuJoCo data
            tau: Torques for controlled joints [3]
        """
        data.ctrl[:] = 0.0
        for i, motor_id in enumerate(self.motor_ids):
            data.ctrl[motor_id] = tau[i]

    def print_info(self):
        """Print robot configuration information"""
        print("=" * 60)
        print("🤖 Robot Configuration")
        print("=" * 60)
        print(f"Controlled joints: {', '.join(self.get_joint_names())}")
        print(f"Joint qpos IDs: {self.controlled_joint_ids}")
        print(f"Motor actuator IDs: {self.motor_ids}")
        print(f"Motor names: {', '.join(self.motor_names)}")
        print("=" * 60)


def setup_robot(model):
    """
    Setup robot interface
    
    Args:
        model: MuJoCo model
    
    Returns:
        RobotInterface instance
    """
    robot = RobotInterface(model)
    robot.print_info()
    return robot
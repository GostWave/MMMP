import os

import numpy as np
import pybullet as p
import pybullet_data
from robots.robot import Robot
from utils.ik_utils import IK_info, calculate_ik, calculate_closest_ik
import utils.pb_conf_utils as pb_conf
from utils.pb_conf_utils import wait_for_duration
from utils.pb_joint_utils import set_joint_positions, get_movable_joints
from utils.pb_link_utils import link_from_name
from scipy.spatial.transform import Rotation as R

class Aubo(Robot):
    AUBO_URDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/aubo_i7/aubo_i7.urdf"))
    MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/aubo_i7"))
    AUBO_INFO = IK_info(base_link='base_link', ee_link='wrist3_Link', free_joints=['wrist3_joint'])

    def load(self):
        p.setAdditionalSearchPath(Aubo.MODELS_DIR)
        result = super().load()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        return result

    def __init__(self, fixed_base=True, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1), scale=1.0):
        super().__init__(Aubo.AUBO_URDF, fixed_base, base_position, base_orientation, scale)
        self.tool_link = link_from_name(self.r_id, 'wrist3_Link')
        print(f"Tool link: {self.tool_link}")
        self.joints = get_movable_joints(self.r_id)
        self.arm_joints = self.joints
        self.c_space = self.get_config_space(self.joints)
        self.arm_c_space = self.c_space
        self.dimension = len(self.joints)
        self.arm_dimension = len(self.arm_joints)

        self.dh_params = [
            {"a": 0, "d": 0.122, "alpha": 0, "theta": 3.14159},
            {"a": 0, "d": 0.1215, "alpha": -1.5708, "theta": -1.5708},
            {"a": 0.368, "d": 0, "alpha": 3.14159, "theta": 0},
            {"a": 0.316, "d": 0, "alpha": 3.14159, "theta": -1.5708},
            {"a": 0, "d": 0.1025, "alpha": -1.5708, "theta": 0},
            {"a": 0, "d": 0.094, "alpha": 1.5708, "theta": 0}
        ]

        os.system('cls' if os.name == 'nt' else 'clear')

    # GETTERS
    def get_arm_joints(self):
        return get_movable_joints(self.r_id)

    def get_arm_pose(self):
        return super().get_pose(self.arm_joints)

    # SETTERS
    def set_arm_pose(self, pose):
        return super().set_pose(self.arm_joints, pose)

    def set_arm_in_standby(self):
        standby_pose = (0, 0, 0, 0, 0, 0)
        self.set_arm_pose(standby_pose)

    def execute_arm_motion(self, arm_path):
        for q in arm_path:
            self.set_arm_pose(q)
            wait_for_duration(0.05)

    # FORWARD KINEMATICS
    def modified_homogeneous_transform(self, a, d, alpha, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -sa*d],
            [st*sa, ct*sa, ca, ca*d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, q):
        base_rot = R.from_quat(self.base_orientation).as_matrix()
        T_base = np.eye(4)
        T_base[:3, :3] = base_rot
        T_base[:3, 3] = self.base_position

        T = T_base
        for i, params in enumerate(self.dh_params):
            T_i = self.modified_homogeneous_transform(params["a"], params["d"], params["alpha"], q[i])
            T = np.matmul(T, T_i)
        return T

    def position_from_fk(self, q):
        return self.forward_kinematics(q)[:3, 3]

    def positions_from_fk(self, q):
        positions = []
        base_rot = R.from_quat(self.base_orientation).as_matrix()
        T_base = np.eye(4)
        T_base[:3, :3] = base_rot
        T_base[:3, 3] = self.base_position

        T = T_base
        for i, params in enumerate(self.dh_params):
            T_i = self.modified_homogeneous_transform(params["a"], params["d"], params["alpha"], q[i])
            T = np.matmul(T, T_i)
            positions.append(T[:3, 3])
        return positions

    def orientation_from_fk(self, q):
        return self.forward_kinematics(q)[:3, :3]

    # INVERSE KINEMATICS
    def solve_arm_ik(self, targetPose):
        return calculate_ik(self.r_id, self.tool_link, targetPose)

    def solve_closest_arm_ik(self, targetPose, initialJointPositions):
        return calculate_closest_ik(self.r_id, self.tool_link, targetPose, initialJointPositions)

    def solve_ik(self, target_position, target_orientation, default_joint_positions):
        lower = [-2 * np.pi] * 6
        upper = [ 2 * np.pi] * 6
        rest = list(default_joint_positions)
        best = None
        for _ in range(200):
            joint_poses = p.calculateInverseKinematics(
                self.r_id,
                endEffectorLinkIndex=self.tool_link,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=lower,
                upperLimits=upper,
                jointRanges=[4 * np.pi] * 6,
                restPoses=rest,
                jointDamping=[0.1] * 6,
                maxNumIterations=2000,
                residualThreshold=1e-4
            )
            result = list(joint_poses[:6])
            for i in range(6):
                while result[i] > upper[i]:
                    result[i] -= 2 * np.pi
                while result[i] < lower[i]:
                    result[i] += 2 * np.pi
            result = tuple(result)
            if all(lower[i] <= result[i] <= upper[i] for i in range(6)):
                return result
            if best is None:
                best = result
            rest = list(np.random.uniform(lower, upper))
        return best if best is not None else tuple(joint_poses[:6])

    # DISTANCE METRICS
    def distance_metric(self, q1, q2):
        return self.task_space_pose_distance(q1, q2)

    def angular_distance_metric(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))

    def task_space_pose_distance(self, q1, q2):
        pos1 = self.positions_from_fk(q1)
        pos2 = self.positions_from_fk(q2)
        return sum(np.linalg.norm(pos1[i] - pos2[i]) for i in range(len(pos1)))

    def task_space_position_distance(self, q1, q2):
        return np.linalg.norm(self.position_from_fk(q1) - self.position_from_fk(q2))

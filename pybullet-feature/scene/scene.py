import pybullet as p
import pybullet_data
import time
import numpy as np

physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")

scene_id = p.loadURDF("figure/scene.urdf", flags=p.URDF_USE_SELF_COLLISION)

wall_1_id = p.loadURDF("figure/wall_scene_1.urdf", basePosition = [0.45, -1.0675, 0.975], baseOrientation = p.getQuaternionFromEuler([0,0,1.57]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE)
wall_2_id = p.loadURDF("figure/wall_scene_2.urdf", basePosition = [-0.55, -0.62, 0.5], baseOrientation = p.getQuaternionFromEuler([0,1.57,0]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE)
wall_3_id = p.loadURDF("figure/wall_scene_2.urdf", basePosition = [-0.3, 0.62, 0.5], baseOrientation = p.getQuaternionFromEuler([0,1.57,0]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE)

robot_1_id = p.loadURDF("aubo_i7/aubo_i7_gripper.urdf", [-0.21, 0, 0.75], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
robot_2_id = p.loadURDF("aubo_i7/aubo_i7_gripper.urdf", [0.21, 0.21, 0.75], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
robot_3_id = p.loadURDF("aubo_i7/aubo_i7_gripper.urdf", [0.21, -0.21, 0.75], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

robots = [robot_1_id, robot_2_id, robot_3_id]

for robot_id in robots:
    num_joints = p.getNumJoints(robot_id)
    for joint_index in range(num_joints):
        p.setJointMotorControl2(bodyIndex=robot_id,
                                jointIndex=joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0,
                                force=500)

wall_1 = p.addUserDebugParameter("Wall 1", -1.1, 2, -1.0675)
wall_2 = p.addUserDebugParameter("Wall 2", 0.5, 1, 0.5)
wall_3 = p.addUserDebugParameter("Wall 3", 0.5, 1, 0.5)

while True:

    if not p.isConnected():
        break

    val_1 = p.readUserDebugParameter(wall_1)
    val_2 = p.readUserDebugParameter(wall_2)
    val_3 = p.readUserDebugParameter(wall_3)

    p.resetBasePositionAndOrientation(
        wall_1_id, 
        [0.45, val_1, 0.975], 
        p.getQuaternionFromEuler([0, 0, 1.57])
    )

    p.resetBasePositionAndOrientation(
        wall_2_id, 
        [-0.55, -0.62, val_2], 
        p.getQuaternionFromEuler([0, 1.57, 0])
    )

    p.resetBasePositionAndOrientation(
        wall_3_id, 
        [-0.3, 0.62, val_3], 
        p.getQuaternionFromEuler([0, 1.57, 0])
    )

    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()

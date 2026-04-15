import pybullet as p
import pybullet_data
import time
import json
import numpy as np

with open('scene/config/config.json', 'r') as f:
    cfg = json.load(f)

# параметры для cube_spacing в config.json
# 0.45 - близко
# 0.7 - свободно 
# 1.2 - разнесены
sp = cfg["cube_spacing"] / 2.0
table_offset = 0.8
dist = sp + table_offset
# dist = cfg["table_distance"]
wall_z_1 = cfg["wall_z_1"]
wall_z_2 = cfg["wall_z_2"]
wall_x_1 = cfg["wall_x_1"]
wall_x_2 = cfg["wall_x_2"]
wall_x_3 = cfg["wall_x_3"]

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")

table_configs = [
    {"file": "figure/table_1.urdf", "pos": [ dist,  0, 0], "ori": [0, 0, 1.57]},
    {"file": "figure/table_2.urdf", "pos": [ 0,  dist, 0], "ori": [0, 0, 0]},
    {"file": "figure/table_3.urdf", "pos": [-dist,  0, 0], "ori": [0, 0, 1.57]},
    {"file": "figure/table_4.urdf", "pos": [ 0, -dist, 0], "ori": [0, 0, 0]}
]

for t in table_configs:
    p.loadURDF(t["file"], t["pos"], p.getQuaternionFromEuler(t["ori"]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

cube_positions = [
    [-sp, 0, cfg["cube_z_height"]],
    [ 0.21, sp, cfg["cube_z_height"]],
    [ 0.21, -sp, cfg["cube_z_height"]]
]

robots = []
for pos in cube_positions:
    p.loadURDF("figure/cube_1.urdf", pos, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

    robot_id = p.loadURDF("aubo_i7/aubo_i7_gripper.urdf", 
                          [pos[0], pos[1], cfg["robot_z_height"]], 
                          useFixedBase=True, 
                          flags=p.URDF_USE_SELF_COLLISION)
    robots.append(robot_id)

wall_1_id = p.loadURDF("figure/wall_scene_1.urdf", [wall_x_1, -dist, wall_z_1], p.getQuaternionFromEuler([0,0,1.57]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE)
wall_2_id = p.loadURDF("figure/wall_scene_2.urdf", [wall_x_2, -dist+0.4, wall_z_2], p.getQuaternionFromEuler([0,1.57,0]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE)
wall_3_id = p.loadURDF("figure/wall_scene_2.urdf", [wall_x_3, dist-0.41, wall_z_2], p.getQuaternionFromEuler([0,1.57,0]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE)

for robot_id in robots:
    num_joints = p.getNumJoints(robot_id)
    for joint_index in range(num_joints):
        p.setJointMotorControl2(bodyIndex=robot_id,
                                jointIndex=joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0,
                                force=500)

wall_1 = p.addUserDebugParameter("Wall 1", -(dist +0.5), (dist +0.5), -dist)
wall_2 = p.addUserDebugParameter("Wall 2", wall_z_2, 1, wall_z_2)
wall_3 = p.addUserDebugParameter("Wall 3", wall_z_2, 1, wall_z_2)

while True:
    if not p.isConnected():
        break

    w1 = p.readUserDebugParameter(wall_1)
    w2 = p.readUserDebugParameter(wall_2)
    w3 = p.readUserDebugParameter(wall_3)

    p.resetBasePositionAndOrientation(wall_1_id, [wall_x_1, w1, wall_z_1], p.getQuaternionFromEuler([0, 0, 1.57]))
    p.resetBasePositionAndOrientation(wall_2_id, [wall_x_2, -dist+0.4, w2], p.getQuaternionFromEuler([0, 1.57, 0]))
    p.resetBasePositionAndOrientation(wall_3_id, [wall_x_3, dist-0.41, w3], p.getQuaternionFromEuler([0, 1.57, 0]))

    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
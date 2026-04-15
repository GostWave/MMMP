import pybullet as p
import pybullet_data
import time
import numpy as np

def camera():
    camera_link_state = p.getLinkState(aubo, camera_link_index)
    camera_position = camera_link_state[0]
    camera_orientation_quat = camera_link_state[1]
    R_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation_quat)).reshape((3,3))
    view_matrix = p.computeViewMatrix(
        camera_position,
        camera_position + camera_distance * R_matrix @ np.array([0, 0, 1]),
        cameraUpVector=R_matrix @ np.array([0, 1, 0])
    )
    image = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    return image

def control_gripper(angle):
    if main_gripper_joint != -1:
        p.setJointMotorControl2(aubo, main_gripper_joint, p.POSITION_CONTROL, targetPosition=angle, force=200)
    for joint_idx, multiplier in mimic_joints:
        p.setJointMotorControl2(aubo, joint_idx, p.POSITION_CONTROL, targetPosition=angle * multiplier, force=200)

def move_to(target_pos):
    global Kp, end_idx, joint_idx_all, joint_idx_control
    
    end_state = p.getLinkState(aubo, end_idx, 1)
    pos = end_state[0]
    error = target_pos - np.array(pos)

    if np.linalg.norm(error) < 0.001:
        p.setJointMotorControlArray(aubo, joint_idx_control, controlMode=p.VELOCITY_CONTROL, targetVelocities=[0]*len(joint_idx_control))
        return
    
    desired_vel_world = error * Kp
    velocity = np.concatenate([desired_vel_world, [0, 0, 0]])

    states = p.getJointStates(aubo, joint_idx_all)
    pos_all = [state[0] for state in states]
    vel_all = [state[1] for state in states]

    J_lin, J_ang = p.calculateJacobian(aubo, end_idx, [0, 0, 0], pos_all, vel_all, [0]*len(joint_idx_all))
    J = np.vstack([np.array(J_lin), np.array(J_ang)])

    J_control = J[:, :len(joint_idx_control)]
    deque = np.linalg.pinv(J_control) @ velocity
    
    p.setJointMotorControlArray(aubo, joint_idx_control, controlMode=p.VELOCITY_CONTROL, targetVelocities=deque)


physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane_id = p.loadURDF("plane.urdf")
aubo = p.loadURDF("aubo_i7/aubo_i7_gripper.urdf", [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

cube_id = p.loadURDF("figure/cube.urdf", basePosition=[0.2, 0.7, 0.26], 
                     baseOrientation=p.getQuaternionFromEuler([0,0,0]), 
                     useFixedBase=False, globalScaling=0.52, 
                     flags=p.URDF_USE_SELF_COLLISION)
p.changeDynamics(cube_id, -1, lateralFriction=1.0, spinningFriction=0.0, 
                rollingFriction=1.0, contactStiffness=1000.0, contactDamping=1000.0)

can_pos = [0.12, 0.51, 0.52]
can_id = p.loadURDF("figure/soda_can.urdf", can_pos)
p.changeDynamics(can_id, -1, lateralFriction=1.0, rollingFriction=0.01)


num_joints = p.getNumJoints(aubo)
joint_idx_all = []
joint_idx_control = []
link_name_to_index = {}
mimic_joints = []
main_gripper_joint = -1
mimic_joint_data = {
    "robotiq_85_right_knuckle_joint": 1.0,
    "robotiq_85_left_inner_knuckle_joint": 1.0,
    "robotiq_85_right_inner_knuckle_joint": 1.0,
    "robotiq_85_left_finger_tip_joint": -1.0,
    "robotiq_85_right_finger_tip_joint": -1.0}

for i in range(num_joints):
    joint_info = p.getJointInfo(aubo, i)
    link_name = joint_info[12].decode('utf-8')
    joint_name = joint_info[1].decode('utf-8')
    link_name_to_index[link_name] = i
    if joint_name == "robotiq_85_left_knuckle_joint": 
        main_gripper_joint = i
    if joint_name in mimic_joint_data: 
        mimic_joints.append((i, mimic_joint_data[joint_name]))
    if joint_info[2] != p.JOINT_FIXED:
        joint_idx_all.append(i)
        if len(joint_idx_control) < 6:
            joint_idx_control.append(i)

end_idx = link_name_to_index["wrist3_Link"]
grip_1 = link_name_to_index["robotiq_85_left_knuckle_link"]
grip_2 = link_name_to_index["robotiq_85_right_knuckle_link"] 

start_pos = [1.57, 0.0, 1.57, 1.57, 1.57, 0.0]
for i in range(len(start_pos)):
    p.resetJointState(aubo, joint_idx_control[i], start_pos[i])

p.resetJointState(aubo, grip_1, 0.0)
p.resetJointState(aubo, grip_2, 0.0)

for joint_idx, _ in mimic_joints:
    p.resetJointState(aubo, joint_idx, 0.0)

camera_link_index = link_name_to_index["camera_link"]
camera_distance = 3
width, height = 640, 480
proj_matrix = p.computeProjectionMatrixFOV(fov=58, aspect=width/height, nearVal=0.1, farVal=10)

prev_open = 0
prev_close = 0

end_state = p.getLinkState(aubo, end_idx, 1)
pos = end_state[0]
open_btn = p.addUserDebugParameter("Open Gripper", 1, 0, 0)
close_btn = p.addUserDebugParameter("Close Gripper", 1, 0, 0)
Kp = 2.0
x_target = p.addUserDebugParameter("Target X", -1, 1, pos[0])
y_target = p.addUserDebugParameter("Target Y", -1, 1, pos[1])
z_target = p.addUserDebugParameter("Target Z", -1, 1, pos[2])
counter = 0
curr_gripper_pos = 0.0

while True:
    if not p.isConnected():
        break

    p.stepSimulation()
    time.sleep(1./240.)

    if counter % 24 == 0:
        image = camera()
    counter += 1

    open_val = p.readUserDebugParameter(open_btn)
    close_val = p.readUserDebugParameter(close_btn)

    if open_val > prev_open:
        curr_gripper_pos = 0.0
        prev_open = open_val
    if close_val > prev_close:
        curr_gripper_pos = 0.25 
        prev_close = close_val

    control_gripper(curr_gripper_pos)
    
    target_pos = np.array([
        p.readUserDebugParameter(x_target),
        p.readUserDebugParameter(y_target),
        p.readUserDebugParameter(z_target)
    ])
    move_to(target_pos)

    prev_open = open_val
    prev_close = close_val

    qKey = ord('q')
    keys = p.getKeyboardEvents()
    if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
        for i in range(len(start_pos)):
            p.resetJointState(aubo, joint_idx_control[i], start_pos[i])
        p.resetJointState(aubo, grip_1, 0.0)
        p.resetJointState(aubo, grip_2, 0.0)
        for joint_idx, _ in mimic_joints:
            p.resetJointState(aubo, joint_idx, 0.0)
        p.removeAllUserParameters()
        current_pos = p.getLinkState(aubo, end_idx, 1)[0]
        open_btn = p.addUserDebugParameter("Open Gripper", 1, 0, 0)
        close_btn = p.addUserDebugParameter("Close Gripper", 1, 0, 0)
        x_target = p.addUserDebugParameter("Target X", -1, 1, current_pos[0])
        y_target = p.addUserDebugParameter("Target Y", -1, 1, current_pos[1])
        z_target = p.addUserDebugParameter("Target Z", -1, 1, current_pos[2])
        prev_open = 0
        prev_close = 0
        continue

p.disconnect()
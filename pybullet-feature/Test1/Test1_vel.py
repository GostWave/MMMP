# Управление по скоростям (Done)
import pybullet as p
import time
import pybullet_data
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

def force_sensor():
    force_data = p.getJointState(bodyUniqueId = aubo, jointIndex = ft_idx)[2]
    return force_data

physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")

aubo = p.loadURDF("aubo_i7/aubo_i7.urdf", basePosition = [0,0,0], baseOrientation = p.getQuaternionFromEuler([0,0,0]), useFixedBase = True, flags=p.URDF_USE_SELF_COLLISION)

num_joints = p.getNumJoints(aubo)
joint_idx = []
link_name_to_index = {}
for i in range(num_joints):
    joint_info = p.getJointInfo(aubo, i)

    link_name = p.getJointInfo(aubo, i)[12].decode('utf-8')
    link_name_to_index[link_name] = i

    if joint_info[2] != p.JOINT_FIXED:
        joint_idx.append(i)

# Start position
start_pos = [1.57, 0.0, -1.57, 0.0, -1.57, 0.0]
for i in range(len(joint_idx)):
    p.resetJointState(aubo, joint_idx[i], start_pos[i])


# Camera
camera_link_index = link_name_to_index["camera_link"]
camera_distance = 3
width, height = 640, 480
proj_matrix = p.computeProjectionMatrixFOV(fov=58, aspect=width/height, nearVal=0.1, farVal=10)

# Force Sensor 
ft_idx = link_name_to_index["ft_Link"]
p.enableJointForceTorqueSensor(bodyUniqueId = aubo, jointIndex = ft_idx)

end_idx = link_name_to_index["wrist3_Link"]
counter = 0

# Test params
x_lin = p.addUserDebugParameter("VX", -1, 1, 0)
y_lin = p.addUserDebugParameter("VY", -1, 1, 0)
z_lin = p.addUserDebugParameter("VZ", -1, 1, 0)
x_ang = p.addUserDebugParameter("WX", -1, 1, 0)
y_ang = p.addUserDebugParameter("WY", -1, 1, 0)
z_ang = p.addUserDebugParameter("WZ", -1, 1, 0)


while True:

    if not p.isConnected():
        break

    p.stepSimulation()
    time.sleep(1./240.)

    qKey = ord('q')
    keys = p.getKeyboardEvents()
    if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
        p.removeAllUserParameters()
        x_lin = p.addUserDebugParameter("VX", -1, 1, 0)
        y_lin = p.addUserDebugParameter("VY", -1, 1, 0)
        z_lin = p.addUserDebugParameter("VZ", -1, 1, 0)
        x_ang = p.addUserDebugParameter("WX", -1, 1, 0)
        y_ang = p.addUserDebugParameter("WY", -1, 1, 0)
        z_ang = p.addUserDebugParameter("WZ", -1, 1, 0)

    ### CAMERA
    if counter % 24 == 0:
        image = camera()
    counter += 1

    ### FORCE SENSOR
    force_data = force_sensor()

    velo_local = np.array([
        p.readUserDebugParameter(x_lin),
        p.readUserDebugParameter(y_lin),
        p.readUserDebugParameter(z_lin), 
        p.readUserDebugParameter(x_ang),
        p.readUserDebugParameter(y_ang),
        p.readUserDebugParameter(z_ang)])

    if np.linalg.norm(velo_local) < 0.001:
        p.setJointMotorControlArray(aubo, joint_idx, controlMode=p.VELOCITY_CONTROL, targetVelocities=[0]*len(joint_idx))
        continue

    end_state = p.getLinkState(aubo, end_idx)
    end_quat = end_state[1]
    rotation_matrix = np.array(p.getMatrixFromQuaternion(end_quat)).reshape(3,3)

    velo_world = np.concatenate([
    rotation_matrix @ velo_local[:3],
    rotation_matrix @ velo_local[3:]])

    states = p.getJointStates(aubo, joint_idx)
    pos_all = [state[0] for state in states]
    vel_all = [state[1] for state in states]

    J_lin, J_ang = p.calculateJacobian(aubo, end_idx, [0, 0, 0], pos_all, vel_all, [0]*len(joint_idx))
    J = np.vstack([np.array(J_lin), np.array(J_ang)])
    deque = np.linalg.pinv(J) @ velo_world
    
    p.setJointMotorControlArray(aubo, joint_idx, controlMode=p.VELOCITY_CONTROL, targetVelocities=deque)

p.disconnect()
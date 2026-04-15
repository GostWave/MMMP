# Кубик двигать с графиками (Done) 
import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt

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
cube_id = p.loadURDF("figure/cube.urdf", basePosition = [0.8, 0, 0.35], baseOrientation = p.getQuaternionFromEuler([0,0,0]), useFixedBase=False, globalScaling=0.7, flags=p.URDF_USE_SELF_COLLISION)
p.changeDynamics(cube_id, -1, lateralFriction=0.1, spinningFriction=0.0, rollingFriction=1.0, contactStiffness=1000.0, contactDamping=1000.0)

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
start_pos = [0.0, 0.0, 1.57, 1.57, 1.57, 0.0]
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

forces_x = []
forces_y = []
forces_z = []
timestamps = []

start_time = time.time()
speed = 0.5
counter = 0

while True:
    if not p.isConnected():
        break

    p.stepSimulation()
    time.sleep(1./240.)

    current_time = time.time()
    duration = current_time - start_time

    ### CAMERA
    if counter % 24 == 0:
        image = camera()
    counter += 1

    ### FORCE SENSOR
    force_data = force_sensor()
    forces_x.append(force_data[0])
    forces_y.append(force_data[1])
    forces_z.append(force_data[2])
    timestamps.append(duration)

    qKey = ord('q')
    keys = p.getKeyboardEvents()
    if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
        break

    end_state = p.getLinkState(aubo, end_idx)
    end_quat = end_state[1]
    rotation_matrix = np.array(p.getMatrixFromQuaternion(end_quat)).reshape(3,3)
    velo_local = np.array([0, 0, speed, 0, 0, 0])

    if np.linalg.norm(velo_local) < 0.001:
        p.setJointMotorControlArray(aubo, joint_idx, controlMode=p.VELOCITY_CONTROL, targetVelocities=[0]*len(joint_idx))
        continue

    velo_world = np.concatenate([rotation_matrix @ velo_local[:3],rotation_matrix @ velo_local[3:]])
    states = p.getJointStates(aubo, joint_idx)
    pos_all = [state[0] for state in states]
    vel_all = [state[1] for state in states]

    J_lin, J_ang = p.calculateJacobian(aubo, end_idx, [0, 0, 0], pos_all, vel_all, [0]*len(joint_idx))
    J = np.vstack([np.array(J_lin), np.array(J_ang)])
    deque = np.linalg.pinv(J) @ velo_world
    p.setJointMotorControlArray(aubo, joint_idx, controlMode=p.VELOCITY_CONTROL, targetVelocities=deque)

p.disconnect()


forces_x = np.array(forces_x)
forces_y = np.array(forces_y)
forces_z = np.array(forces_z)
timestamps = np.array(timestamps)


fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(timestamps, forces_x, 'r-', linewidth=1.5)
axes[0].set_ylabel('Fx', fontsize=12)
axes[0].set_xlabel('Время (с)', fontsize=12)
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([-1, 10])

axes[1].plot(timestamps, forces_y, 'g-', linewidth=1.5)
axes[1].set_ylabel('Fy', fontsize=12)
axes[1].set_xlabel('Время (с)', fontsize=12)
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([-1, 10])

axes[2].plot(timestamps, forces_z, 'b-', linewidth=1.5)
axes[2].set_ylabel('Fz', fontsize=12)
axes[2].set_xlabel('Время (с)', fontsize=12)
axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([-1, 10])

plt.tight_layout()

filename = f'Test3/force_v_{speed}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()
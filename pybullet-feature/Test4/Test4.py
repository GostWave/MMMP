# Регулятор усилий
import pybullet as p
import time
import pybullet_data
import spatialmath as spm
import spatialmath.base as spmb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def calculate_quaternion_matrix(q: np.ndarray) -> np.ndarray:
    qe = np.array([
        [-q[1],  q[0], -q[3],  q[2]],
        [-q[2],  q[3],  q[0], -q[1]],
        [-q[3], -q[2],  q[1],  q[0]]
    ])

    return qe

def get_error(goal: spm.SE3, transform: spm.SE3) -> np.ndarray:
    error_translation = goal.t - transform.t
    qe = calculate_quaternion_matrix(spm.UnitQuaternion(transform).A)
    return np.concatenate((error_translation, 2*qe @ spm.UnitQuaternion(goal).A))

physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")

aubo = p.loadURDF("aubo_i7/aubo_i7.urdf", basePosition = [0,0,0], baseOrientation = p.getQuaternionFromEuler([0,0,0]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE)
cube_id = p.loadURDF("figure/wall.urdf", basePosition = [0, 0.465, 0.35], baseOrientation = p.getQuaternionFromEuler([0,0,0]), useFixedBase=False, globalScaling=0.7, flags=p.URDF_USE_INERTIA_FROM_FILE)
p.changeDynamics(cube_id, -1, lateralFriction=1.0, spinningFriction=0.0, rollingFriction=0.1, contactStiffness=1000.0, contactDamping=100.0)

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
start_pos = [1.57, 0.0, 1.57, 1.57, 1.57, 0.0]
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

for j in joint_idx:
    p.setJointMotorControl2(aubo, j, p.VELOCITY_CONTROL, force=0)
end_idx = link_name_to_index["wrist3_Link"]
end_state = p.getLinkState(aubo, end_idx, 1)
target_global = np.array(end_state[0]).reshape(3,1)
quat = end_state[1]
rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
target_force = np.array([0, 0, 1]).reshape(3,1)
counter = 0

Kp = np.diag([500, 500, 500, 5, 5, 5])
kv = 50
k = 0.0001

force_x = []
force_y = []
force_z = []
times = []
frames = 0

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
line1, = ax1.plot([], [], 'r-', linewidth=1.5, label='Fx')
line2, = ax2.plot([], [], 'g-', linewidth=1.5, label='Fy')
line3, = ax3.plot([], [], 'b-', linewidth=1.5, label='Fz')

ax1.set_ylabel('Fx (N)')
ax1.set_ylim([-0.5, 2.5])
ax1.grid(True)
ax1.legend()

ax2.set_ylabel('Fy (N)')
ax2.set_ylim([-0.5, 2.5])
ax2.grid(True)
ax2.legend()

ax3.set_xlabel('Время')
ax3.set_ylabel('Fz (N)')
ax3.set_ylim([-0.5, 2.5])
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show(block=False)

while True:
    if not p.isConnected():
        break

    p.stepSimulation()
    time.sleep(1./240.)

    ### CAMERA
    if counter % 100 == 0:
        image = camera()
    counter += 1

    ### FORCE SENSOR
    force_data = np.array(force_sensor())
    force_x.append(force_data[0])
    force_y.append(force_data[1])
    force_z.append(force_data[2])
    times.append(frames)
    frames += 1
    if frames % 10 == 0:
        line1.set_data(times, force_x)
        line2.set_data(times, force_y)
        line3.set_data(times, force_z)
        if len(times) > 0:
            ax1.set_xlim([max(0, times[-1] - 500), times[-1] + 50])
            ax2.set_xlim([max(0, times[-1] - 500), times[-1] + 50])
            ax3.set_xlim([max(0, times[-1] - 500), times[-1] + 50])
        
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    force_state = p.getLinkState(aubo, end_idx, 1)
    position = np.array(force_state[0]).reshape(3,1)
    quaternion = force_state[1]

    states = p.getJointStates(aubo, joint_idx)
    pos_all = [state[0] for state in states]
    vel_all = [state[1] for state in states]

    zero = [0.0] * len(joint_idx)
    G = p.calculateInverseDynamics(aubo, pos_all, zero, zero)
    M = np.array(p.calculateMassMatrix(aubo, pos_all))

    force_rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3,3)
    tr = np.block([
        [force_rotation, position],
        [0, 0, 0, 1]])
    
    J_lin, J_ang = p.calculateJacobian(aubo, end_idx, [0, 0, 0], pos_all, vel_all, [0]*len(joint_idx))
    J = np.vstack([np.array(J_lin), np.array(J_ang)])
    fr_err = force_rotation @ (force_data[:3].reshape(3,1) - target_force)

    target_global[1] -= k * fr_err[1]

    if -0.01<fr_err[1]<0.01:
        target_global[0]+=0.01

    g = np.block([
        [rotation, target_global],
        [0, 0, 0, 1]])
    goal = spm.SE3(g)

    transform = spm.SE3(tr)
    err = get_error(goal, transform)
    velocity = np.array(G) - kv * M @ vel_all + J.T @ Kp @ err
    p.setJointMotorControlArray(aubo, joint_idx, controlMode = p.TORQUE_CONTROL, forces=velocity)

    qKey = ord('q')
    keys = p.getKeyboardEvents()
    if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
        break

plt.ioff()
plt.show()
p.disconnect()
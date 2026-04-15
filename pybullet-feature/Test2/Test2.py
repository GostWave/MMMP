# Слежение за маркером (Done)
import pybullet as p
import time
import pybullet_data
import numpy as np
import cv2

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
    force_data = p.getJointState(bodyUniqueId = aubo, jointIndex = 6)[2]
    return force_data


physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")

# Aruco marker
cube_id = p.loadURDF("figure/cube.urdf", basePosition = [0, -0.45, 0.03], baseOrientation = p.getQuaternionFromEuler([0,0,0]), useFixedBase = False, globalScaling=0.06, flags=p.URDF_USE_SELF_COLLISION)
marker_file = 'figure/box_aruco_marker_id_131_with_border.png'
texture_id = p.loadTexture(marker_file)
p.changeVisualShape(cube_id, -1, textureUniqueId=texture_id)

aubo = p.loadURDF("aubo_i7/aubo_i7.urdf", basePosition = [0,0,0], baseOrientation = p.getQuaternionFromEuler([0,0,0]), useFixedBase = True, flags=p.URDF_USE_SELF_COLLISION)

# Start position
start_pos = [0.0, 1.57, 0.0, -1.57, 0.0, -1.57, 0.0]
for i in range(len(start_pos)):
    p.resetJointState(aubo, i, start_pos[i])

link_name_to_index = {}
num_joints = p.getNumJoints(aubo)
joint_idx = []
link_name_to_index = {}
for i in range(num_joints):
    joint_info = p.getJointInfo(aubo, i)

    link_name = p.getJointInfo(aubo, i)[12].decode('utf-8')
    link_name_to_index[link_name] = i

    if joint_info[2] != p.JOINT_FIXED:
        joint_idx.append(i)
end_idx = link_name_to_index["wrist3_Link"]

# Camera
camera_link_index = link_name_to_index["camera_link"]
camera_distance = 3
width, height = 640, 480
proj_matrix = p.computeProjectionMatrixFOV(fov=58, aspect=width/height, nearVal=0.1, farVal=10)

# Force Sensor 
ft_idx = link_name_to_index["ft_Link"]
p.enableJointForceTorqueSensor(bodyUniqueId = aubo, jointIndex = ft_idx)

# Test params
Kp = 3
marker_pos = [0, -0.45, 0.1]
marker_speed = 0.05
target =[0, 0, 0.5]
counter = 0

while True:

    if not p.isConnected():
        break
    
    # Marker control
    key = cv2.waitKey(1) & 0xFF    
    if key == ord('w') or key == ord('W'):
        marker_pos[1] += marker_speed
    elif key == ord('s') or key == ord('S'):
        marker_pos[1] -= marker_speed
    elif key == ord('a') or key == ord('A'):
        marker_pos[0] -= marker_speed
    elif key == ord('d') or key == ord('D'):
        marker_pos[0] += marker_speed
    p.resetBasePositionAndOrientation(cube_id, marker_pos, p.getQuaternionFromEuler([0,0,0]))

    p.stepSimulation()
    time.sleep(1./240.)

    end_state = p.getLinkState(aubo, camera_link_index)
    end_pos = end_state[0]

    ### FORCE SENSOR
    force_data = force_sensor()

    ### CAMERA
    if counter % 2 == 0:
        image = camera()
    counter += 1
    
    if image is not None and len(image) > 2:
        img_rgb = np.array(image[2])
        img_rgb = img_rgb[:, :, :3]
        img_rgb = img_rgb[:, :, [2, 1, 0]]

        img_display = img_rgb.copy()
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected_candidates = detector.detectMarkers(img_display)

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(img_display, corners)

            aruco_corners = corners[0][0]
            aruco_cx = np.mean(aruco_corners[:, 0])
            aruco_cy = np.mean(aruco_corners[:, 1])

            img_center_x, img_center_y = width//2, height//2
            img_error_x = (img_center_x - aruco_cx) / width
            img_error_y = (aruco_cy - img_center_y) / height

            proj = np.array(proj_matrix).reshape(4, 4)
            cx = width / 2
            cy = height / 2
            fx = proj[0][0] * cx
            fy = proj[1][1] * cy
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            marker_size = 0.06
            object_points = np.array([
                [-marker_size/2,  marker_size/2, 0],  
                [ marker_size/2,  marker_size/2, 0],  
                [ marker_size/2, -marker_size/2, 0], 
                [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)
            _, _, tvec = cv2.solvePnP(object_points, corners[0][0], camera_matrix, None)
            tvec = np.array(tvec).T
            img_error = (target - tvec)

            cv2.circle(img_display, (int(aruco_cx), int(aruco_cy)), 5, (0, 0, 255), -1)
            cv2.line(img_display, (int(aruco_cx), int(aruco_cy)), (img_center_x, img_center_y), (255, 0, 0), 2)

            cv2.putText(img_display, f"Err X: {img_error[0][0]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Err Y: {img_error[0][1]:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Err Z: {img_error[0][2]:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if abs(img_error[0][0])>=0.01 or abs(img_error[0][1])>=0.01 or abs(img_error[0][2])>=0.01:
                velik = Kp * img_error
                vel_x = velik[0][0]
                vel_y = -velik[0][1]
                vel_z = velik[0][2]

                velocity = np.array([vel_x, vel_y, vel_z, 0, 0, 0])

                states = p.getJointStates(aubo, joint_idx)
                pos_all = [state[0] for state in states]
                vel_all = [state[1] for state in states]

                J_lin, J_ang = p.calculateJacobian(aubo, end_idx, [0, 0, 0], pos_all, vel_all, [0]*len(joint_idx))
                J = np.vstack([np.array(J_lin), np.array(J_ang)])
                deque = np.linalg.pinv(J) @ velocity

            else:
                deque = np.array([0, 0, 0, 0, 0, 0])
                cv2.putText(img_display, "DONE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            velocity = np.array([0, 0, 0.5, 0, 0, 0.5])

            states = p.getJointStates(aubo, joint_idx)
            pos_all = [state[0] for state in states]
            vel_all = [state[1] for state in states]

            J_lin, J_ang = p.calculateJacobian(aubo, end_idx, [0, 0, 0], pos_all, vel_all, [0]*len(joint_idx))
            J = np.vstack([np.array(J_lin), np.array(J_ang)])
            deque = np.linalg.pinv(J) @ velocity
            
            cv2.putText(img_display, "No marker detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        p.setJointMotorControlArray(aubo, joint_idx, controlMode=p.VELOCITY_CONTROL, targetVelocities=deque)
        cv2.imshow("Aruco Detection", img_display)

p.disconnect()
cv2.destroyAllWindows()

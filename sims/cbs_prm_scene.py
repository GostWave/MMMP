import os
import time
import numpy as np
import pybullet as p

from utils.pb_data_utils import save_gif, save_video
from planners.prm.pybullet.env import Environment
from planners.prm.pybullet.decoupled.cbsprm import CBSPRM
from robots.aubo import Aubo
from utils.pb_conf_utils import add_data_path, connect, disconnect, pause_sim, set_camera_pose

FIGURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'pybullet-feature', 'figure')
PEDESTAL_HEIGHT = 0.80  # высота верхней поверхности пьедесталов cube_1/2/3


POINT_A = (-0.30,  0.25, PEDESTAL_HEIGHT + 0.40)   # перед-центр
POINT_B = ( 0.50, 0.30, PEDESTAL_HEIGHT + 0.40)   # право
POINT_C = (-0.30,  0.10, PEDESTAL_HEIGHT + 0.40)   # лево-зад
POINT_D = (-0.20, -0.35, PEDESTAL_HEIGHT + 0.40)   # рядом с C, смещение по Y


# POINT_A = (-0.60,  0.4, PEDESTAL_HEIGHT + 0.50)   # перед-центр
# POINT_B = ( 0.7, 0.4, PEDESTAL_HEIGHT + 0.40)   # право
# POINT_C = (0.15,  -0.50, PEDESTAL_HEIGHT + 0.40)   # лево-зад

def load_scene():
    scene_id = p.loadURDF(
        os.path.join(FIGURE_DIR, 'scene.urdf'),
        flags=p.URDF_USE_SELF_COLLISION
    )
    wall_1_id = p.loadURDF(
        os.path.join(FIGURE_DIR, 'wall_scene_1.urdf'),
        basePosition=[0.45, -1.0675, 0.975],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 1.57]),
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
    )
    wall_2_id = p.loadURDF(
        os.path.join(FIGURE_DIR, 'wall_scene_2.urdf'),
        basePosition=[-0.55, -0.62, 0.5],
        baseOrientation=p.getQuaternionFromEuler([0, 1.57, 0]),
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
    )
    wall_3_id = p.loadURDF(
        os.path.join(FIGURE_DIR, 'wall_scene_2.urdf'),
        basePosition=[-0.3, 0.62, 0.5],
        baseOrientation=p.getQuaternionFromEuler([0, 1.57, 0]),
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
    )
    return [scene_id, wall_1_id, wall_2_id, wall_3_id]


def main():
    connect(use_gui=True)
    p.addUserDebugLine([0,0,0], [2,0,0], [1,0,0], lineWidth=5)  # X — красная
    p.addUserDebugLine([0,0,0], [0,2,0], [0,1,0], lineWidth=5)  # Y — зелёная
    p.addUserDebugLine([0,0,0], [0,0,2], [0,0,1], lineWidth=5)  # Z — синяя
    add_data_path()
    set_camera_pose(camera_point=[0, -1.2, 1.2])

    aubo1 = Aubo(base_position=(-0.21, 0, 0.80), base_orientation=(0, 0, 1, 0))
    aubo2 = Aubo(base_position=(0.21, 0.21, 0.80), base_orientation=(0, 0, 1, 0))
    aubo3 = Aubo(base_position=(0.21, -0.21, 0.80), base_orientation=(0, 0, -0.5, 0.866))

    ground = p.loadURDF("plane.urdf")
    scene_objects = load_scene()
    scene_id = scene_objects[0]
    obstacles = [ground] + scene_objects

    # Disable collision between each robot and its mounting cube
    cube_link_indices = {}
    for i in range(p.getNumJoints(scene_id)):
        link_name = p.getJointInfo(scene_id, i)[12].decode('utf-8')
        if link_name in ('cube_1', 'cube_2', 'cube_3'):
            cube_link_indices[link_name] = i
    num_links = p.getNumJoints(aubo1.r_id)
    for robot, cube_name in [(aubo1, 'cube_1'), (aubo2, 'cube_2'), (aubo3, 'cube_3')]:
        cube_link = cube_link_indices[cube_name]
        for link_idx in range(-1, num_links):
            p.setCollisionFilterPair(robot.r_id, scene_id, link_idx, cube_link, enableCollision=0)

    reference_joint_positions = [0, 0, 0, 0, 0, 0]
    aubo1.set_arm_pose(reference_joint_positions)
    aubo2.set_arm_pose(reference_joint_positions)
    aubo3.set_arm_pose(reference_joint_positions)

    # Calculate and set start pose of arm 1
    tool_position_start1 = POINT_C  # aubo1 start: C
    tool_orientation_start1 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    arm1_start = aubo1.solve_closest_arm_ik((tool_position_start1, tool_orientation_start1), reference_joint_positions)
    arm1_start = tuple(round(c, 3) for c in arm1_start)

    if arm1_start is not None:
        pause_sim('Show start pose for first Aubo... (Press Enter to continue)')
        # print("IK Configuration for Aubo 1 start pose:", arm1_start)
        aubo1.set_arm_pose(arm1_start)
        print(f"Start q for Aubo 1: {arm1_start}")
        print(f"Start position for Aubo 1: {aubo1.position_from_fk(arm1_start)}")
    else:
        print("No IK solution found for start pose of second Aubo.")


    # Calculate and set goal pose of arm 1
    tool_position_goal1 = POINT_A  # aubo1 goal: A
    tool_orientation_goal1 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    arm1_goal = aubo1.solve_ik(tool_position_goal1, tool_orientation_goal1, reference_joint_positions)
    arm1_goal = tuple(round(c, 3) for c in arm1_goal)
    
    if arm1_goal is not None:
        pause_sim('Show goal pose for first Aubo... (Press Enter to continue)')
        aubo1.set_arm_pose(arm1_goal)
        print(f"Goal q for Aubo 1: {arm1_goal}")
        print(f"Goal position for Aubo 1: {aubo1.position_from_fk(arm1_goal)}")
    else:
        print("No IK solution found for goal pose of first Aubo.")


    # Calculate and set start of arm 2
    tool_position_start2 = POINT_A  # aubo2 start: A
    tool_orientation_start2 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    arm2_start = aubo2.solve_ik(tool_position_start2, tool_orientation_start2, reference_joint_positions)
    arm2_start = tuple(round(c, 3) for c in arm2_start)
    
    if arm2_start is not None:
        pause_sim('Show start pose for second Aubo... (Press Enter to continue)')
        aubo2.set_arm_pose(arm2_start)
        print(f"Start q for Aubo 2: {arm2_start}")
        print(f"Start position for Aubo 2: {aubo2.position_from_fk(arm2_start)}")
    else:
        print("No IK solution found for start pose of second Aubo.")


    # Calculate and set goal of arm 2
    tool_position_goal2 = POINT_B  # aubo2 goal: B
    tool_orientation_goal2 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    arm2_goal = aubo2.solve_ik(tool_position_goal2, tool_orientation_goal2, arm2_start)
    arm2_goal = tuple(round(c, 3) for c in arm2_goal)
    
    if arm2_goal is not None:
        pause_sim('Show goal pose for second Aubo... (Press Enter to continue)')
        aubo2.set_arm_pose(arm2_goal)
        print(f"Goal q for Aubo 2: {arm2_goal}")
        print(f"Goal position for Aubo 2: {aubo2.position_from_fk(arm2_goal)}")
    else:
        print("No IK solution found for goal pose of second Aubo.")


    # Calculate and set start of arm 3
    tool_position_start3 = POINT_C  # aubo3 start: C
    tool_orientation_start3 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    reference_joint_positions_3 = [0, -np.pi/3, np.pi/2, -np.pi/6, 0, 0]
    arm3_start = aubo3.solve_ik(tool_position_start3, tool_orientation_start3, reference_joint_positions_3)
    arm3_start = tuple(round(c, 3) for c in arm3_start)

    if arm3_start is not None:
        pause_sim('Show start pose for third Aubo... (Press Enter to continue)')
        aubo3.set_arm_pose(arm3_start)
        print(f"Start q for Aubo 3: {arm3_start}")
        print(f"Start position for Aubo 3: {aubo3.position_from_fk(arm3_start)}")
    else:
        print("No IK solution found for start pose of third Aubo.")


    # Calculate and set goal of arm 3
    tool_position_goal3 = POINT_D  # aubo3 goal: D (рядом с C)
    tool_orientation_goal3 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    arm3_goal = aubo3.solve_ik(tool_position_goal3, tool_orientation_goal3, reference_joint_positions_3)
    arm3_goal = tuple(round(c, 3) for c in arm3_goal)

    if arm3_goal is not None:
        pause_sim('Show goal pose for third Aubo... (Press Enter to continue)')
        aubo3.set_arm_pose(arm3_goal)
        print(f"Goal q for Aubo 3: {arm3_goal}")
        print(f"Goal position for Aubo 3: {aubo3.position_from_fk(arm3_goal)}")
    else:
        print("No IK solution found for goal pose of third Aubo.")


    agents = [
            {"name": "agent1", "start": arm1_start, "goal": arm1_goal, "model": aubo1},
            {"name": "agent2", "start": arm2_start, "goal": arm2_goal, "model": aubo2},
            {"name": "agent3", "start": arm3_start, "goal": arm3_goal, "model": aubo3}
        ]
    
    obstacles = [ground]+scene_objects

    pause_sim('Load environment and reset poses to start... (Press Enter to continue)')
    env = Environment(agents, obstacles)

    pause_sim('Learn?')
    start_time = time.time()
    directory = os.path.dirname(__file__)
    roadmap_file = os.path.join(directory, '../res/images/cbs_prm_scene_roadmap.csv')
    all_robot_files = all(os.path.exists(roadmap_file.replace('.csv', f'_robot_{i}.csv')) for i in range(len(agents)))
    prm = CBSPRM(env, load_roadmap=roadmap_file if all_robot_files else None, maxdist=0.1, k1=50, k2=20, build_type='n', prm_type='degree', n=200, t=10, time_step=0.01, local_step=0.02)
    if not all_robot_files:
        prm.save_roadmaps(roadmap_file)
    learn_duration = time.time()-start_time
    print(f"Learning duration: {learn_duration}")
    # print(f"Edges: {prm.edge_dicts}")
    print(f'Average degree: {np.mean(list(len(prm.edge_dicts[0][n_id]) for n_id in prm.edge_dicts[0].keys()))}')
    aubo1.set_arm_pose(arm1_start)
    aubo2.set_arm_pose(arm2_start)
    aubo3.set_arm_pose(arm3_start)

    pause_sim('Query?')
    start_time = time.time()
    paths, l_t, q_t = prm.query()
    query_duration = time.time()-start_time
    if not paths:
        print("Solution not found")
        pause_sim('Disconnect?')
        disconnect()
        return
    print(f"Solution: {paths}")
    print(f"Query duration: {query_duration}")
    aubo1.set_arm_pose(arm1_start)
    aubo2.set_arm_pose(arm2_start)
    aubo3.set_arm_pose(arm3_start)

    frames = None
    # # pause_sim('execute joint motion?')
    # # frames = env.execute_joint_motion_capturing_frames(paths)
    # # frames = env.execute_joint_motion(paths)
    # pause_sim('execute interpolated motion?')
    # aubo1.set_arm_pose(arm1_start)
    # aubo2.set_arm_pose(arm2_start)
    # # env.execute_interpolated_motion(prm, paths, t_final=5.0)
    # frames, _ = env.execute_interpolated_motion_capturing_frames(prm, paths, t_final=5.0)
    
    pause_sim('execute smooth motion?')
    aubo1.set_arm_pose(arm1_start)
    aubo2.set_arm_pose(arm2_start)
    aubo3.set_arm_pose(arm3_start)
    # env.execute_smooth_motion(prm, paths, t_final=5.0, effort_factor=0.8)
    frames, _ = env.execute_smooth_motion_capturing_frames(prm, paths, t_final=5.0, effort_factor=0.8)
    pause_sim('Disconnect?')
    disconnect()

    if frames is not None:
        if input('Save video? (y/N): ').lower() == 'y':
            now = time.strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(os.path.dirname(__file__), '..', 'res', 'videos')
            filename = f'CBSPRM_{now}.mp4'
            save_video(frames, directory, filename, fps=30)
        else:
            print("Video not saved.")

        if input('Save GIF? (y/N): ').lower() == 'y':
            now = time.strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(os.path.dirname(__file__), '..', 'res', 'gifs')
            filename = f'CBSPRM_{now}.gif'
            save_gif(frames, directory, filename, duration=33)
        else:
            print("GIF not saved.")

if __name__ == "__main__":
    main()

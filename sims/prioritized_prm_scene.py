import os
import time
import numpy as np
import pybullet as p
from itertools import permutations

from utils.pb_data_utils import save_gif, save_video
from planners.prm.pybullet.env import Environment
from planners.prm.pybullet.decoupled.prioritized_prm import PrioritizedPRM
from robots.aubo import Aubo
from utils.pb_conf_utils import add_data_path, connect, disconnect, pause_sim, set_camera_pose

FIGURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'pybullet-feature', 'figure')
PEDESTAL_HEIGHT = 0.80


EE_HEIGHT = PEDESTAL_HEIGHT + 0.40

# Раздельные зоны — смещение от базы аналогично aubo1 (Δx≈-0.09, Δy≈±0.10..0.25)
# aubo1 (база -0.21, 0):     левая зона
AUBO1_START = (-0.30,  0.10, EE_HEIGHT)
AUBO1_GOAL  = (-0.30,  0.25, EE_HEIGHT)

# aubo2 (база 0.21, 0.21):   передняя зона, движение по X
AUBO2_START = ( 0.12,  0.31, EE_HEIGHT)
AUBO2_GOAL  = ( 0.27,  0.31, EE_HEIGHT)

# aubo3 (база 0.21, -0.21):  задняя зона, движение по X
AUBO3_START = ( 0.12, -0.31, EE_HEIGHT)
AUBO3_GOAL  = ( 0.27, -0.31, EE_HEIGHT)


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

    aubo1 = Aubo(base_position=(-0.21, 0, 0.751), base_orientation=(0, 0, 1, 0))
    aubo2 = Aubo(base_position=(0.21, 0.21, 0.751), base_orientation=(0, 0, 1, 0))
    aubo3 = Aubo(base_position=(0.21, -0.21, 0.751), base_orientation=(0, 0, 1, 0))

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

    ee_orientation = p.getQuaternionFromEuler([np.radians(180), 0, 0])

    # Calculate and set start pose of arm 1
    arm1_start = aubo1.solve_ik(AUBO1_START, ee_orientation, reference_joint_positions)
    arm1_start = tuple(round(c, 3) for c in arm1_start)

    if arm1_start is not None:
        pause_sim('Show start pose for first Aubo... (Press Enter to continue)')
        aubo1.set_arm_pose(arm1_start)
        print(f"Start q for Aubo 1: {arm1_start}")
        print(f"Start position for Aubo 1: {aubo1.position_from_fk(arm1_start)}")
    else:
        print("No IK solution found for start pose of Aubo 1.")


    # Calculate and set goal pose of arm 1
    arm1_goal = aubo1.solve_ik(AUBO1_GOAL, ee_orientation, arm1_start)
    arm1_goal = tuple(round(c, 3) for c in arm1_goal)

    if arm1_goal is not None:
        pause_sim('Show goal pose for first Aubo... (Press Enter to continue)')
        aubo1.set_arm_pose(arm1_goal)
        print(f"Goal q for Aubo 1: {arm1_goal}")
        print(f"Goal position for Aubo 1: {aubo1.position_from_fk(arm1_goal)}")
    else:
        print("No IK solution found for goal pose of first Aubo.")


    # Calculate and set start of arm 2
    arm2_start = aubo2.solve_ik(AUBO2_START, ee_orientation, reference_joint_positions)
    arm2_start = tuple(round(c, 3) for c in arm2_start)

    if arm2_start is not None:
        pause_sim('Show start pose for second Aubo... (Press Enter to continue)')
        aubo2.set_arm_pose(arm2_start)
        print(f"Start q for Aubo 2: {arm2_start}")
        print(f"Start position for Aubo 2: {aubo2.position_from_fk(arm2_start)}")
    else:
        print("No IK solution found for start pose of Aubo 2.")


    # Calculate and set goal of arm 2
    arm2_goal = aubo2.solve_ik(AUBO2_GOAL, ee_orientation, arm2_start)
    arm2_goal = tuple(round(c, 3) for c in arm2_goal)

    if arm2_goal is not None:
        pause_sim('Show goal pose for second Aubo... (Press Enter to continue)')
        aubo2.set_arm_pose(arm2_goal)
        print(f"Goal q for Aubo 2: {arm2_goal}")
        print(f"Goal position for Aubo 2: {aubo2.position_from_fk(arm2_goal)}")
    else:
        print("No IK solution found for goal pose of Aubo 2.")


    # Calculate and set start of arm 3
    arm3_start = aubo3.solve_ik(AUBO3_START, ee_orientation, reference_joint_positions)
    arm3_start = tuple(round(c, 3) for c in arm3_start)

    if arm3_start is not None:
        pause_sim('Show start pose for third Aubo... (Press Enter to continue)')
        aubo3.set_arm_pose(arm3_start)
        print(f"Start q for Aubo 3: {arm3_start}")
        print(f"Start position for Aubo 3: {aubo3.position_from_fk(arm3_start)}")
    else:
        print("No IK solution found for start pose of Aubo 3.")


    # Calculate and set goal of arm 3
    arm3_goal = aubo3.solve_ik(AUBO3_GOAL, ee_orientation, arm3_start)
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

    obstacles = [ground] + scene_objects

    pause_sim('Load environment and reset poses to start... (Press Enter to continue)')
    env = Environment(agents, obstacles)

    pause_sim('Learn?')
    start_time = time.time()
    directory = os.path.dirname(__file__)
    roadmap_file = os.path.join(directory, '../res/images/prioritized_prm_scene_roadmap.csv')
    all_robot_files = all(os.path.exists(roadmap_file.replace('.csv', f'_robot_{i}.csv')) for i in range(len(agents)))
    prm = PrioritizedPRM(env, load_roadmap=roadmap_file if all_robot_files else None, maxdist=0.1, k1=50, k2=20, build_type='n', prm_type='degree', n=200, t=10, time_step=0.01, local_step=0.02)
    if not all_robot_files:
        prm.save_roadmaps(roadmap_file)
    learn_duration = time.time() - start_time
    print(f"Learning duration: {learn_duration}")
    print(f'Average degree: {np.mean(list(len(prm.edge_dicts[0][n_id]) for n_id in prm.edge_dicts[0].keys()))}')
    aubo1.set_arm_pose(arm1_start)
    aubo2.set_arm_pose(arm2_start)
    aubo3.set_arm_pose(arm3_start)

    pause_sim('Query?')
    start_time = time.time()
    paths = {}
    l_t = q_t = 0
    robot_names = ['aubo1', 'aubo2', 'aubo3']
    for perm in permutations([1, 2, 3]):
        priorities = list(perm)
        order = ['aubo1', 'aubo2', 'aubo3']
        order_str = ' → '.join(robot_names[i] for i in sorted(range(3), key=lambda x: -priorities[x]))
        print(f"\nTrying priorities {priorities}: {order_str}")
        paths, l_t, q_t = prm.query(priorities=priorities)
        if paths:
            print(f"Solution found with priorities {priorities}: {order_str}")
            break
        for r_id in prm.r_ids:
            try:
                prm.delete_start_goal_nodes(r_id)
            except Exception:
                pass
    query_duration = time.time() - start_time
    if not paths:
        print("Solution not found for any priority ordering")
        pause_sim('Disconnect?')
        disconnect()
        return
    print(f"Query duration: {query_duration}")
    aubo1.set_arm_pose(arm1_start)
    aubo2.set_arm_pose(arm2_start)
    aubo3.set_arm_pose(arm3_start)

    frames = None

    pause_sim('execute smooth motion?')
    aubo1.set_arm_pose(arm1_start)
    aubo2.set_arm_pose(arm2_start)
    aubo3.set_arm_pose(arm3_start)
    frames, _ = env.execute_smooth_motion_capturing_frames(prm, paths, t_final=5.0, effort_factor=0.8)
    pause_sim('Disconnect?')
    disconnect()

    if frames is not None:
        if input('Save video? (y/N): ').lower() == 'y':
            now = time.strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(os.path.dirname(__file__), '..', 'res', 'videos')
            filename = f'PrioritizedPRM_{now}.mp4'
            save_video(frames, directory, filename, fps=30)
        else:
            print("Video not saved.")

        if input('Save GIF? (y/N): ').lower() == 'y':
            now = time.strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(os.path.dirname(__file__), '..', 'res', 'gifs')
            filename = f'PrioritizedPRM_{now}.gif'
            save_gif(frames, directory, filename, duration=33)
        else:
            print("GIF not saved.")

if __name__ == "__main__":
    main()

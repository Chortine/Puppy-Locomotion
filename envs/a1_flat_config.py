from real_deployment.base_legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from collections import OrderedDict

import numpy as np
import os, sys

current_root = os.path.dirname(os.path.dirname(__file__))

var_init_pos = [0.0, 0.0, 0.42]
var_fix_base_link = False
var_action_scale = 0.25
var_decimation = 25
var_dt = 0.002
var_relative_action = False
var_control_mode = 'pos'
if var_relative_action:
    var_relative_action = 'pos'  # in relative action case, can only use pos control
    var_action_scale = 0.03


"""
hang_test
walk_test
"""
unique_p = 20.0
unique_d = 0.5

# ===== stiffness ====== #
if unique_p is not None:
    # fore legs
    fore_j0_p = unique_p
    fore_j1_p = unique_p
    fore_j2_p = unique_p

    # hind legs
    hind_j0_p = unique_p
    hind_j1_p = unique_p
    hind_j2_p = unique_p

else:
    # fore legs
    fore_j0_p = 3.0
    fore_j1_p = 1.5
    fore_j2_p = 1.5

    # hind legs
    hind_j0_p = 3.0
    hind_j1_p = 1.5
    hind_j2_p = 1.5

# ===== damping ======= #
if unique_d is not None:
    # fore legs
    fore_j0_d = unique_d
    fore_j1_d = unique_d
    fore_j2_d = unique_d

    # hind legs
    hind_j0_d = unique_d
    hind_j1_d = unique_d
    hind_j2_d = unique_d

else:
    fore_j0_d = 0.01
    fore_j1_d = 0.01
    fore_j2_d = 0.01

    # hind legs
    hind_j0_d = 0.01
    hind_j1_d = 0.01
    hind_j2_d = 0.01

legs_name = ['rr', 'rl', 'fr', 'fl']

obs_mem_len = 4

observation_states_size = OrderedDict({  # the order matters
    # 'sequence_dof_pos': 50 * 12,
    # 'sequence_dof_action': 50 * 12,
    'angular_v': 3 * obs_mem_len,
    'row_pitch': 4 * obs_mem_len,
    'top_commands': 3,
    # 'dof_pos': 12,
    # 'dof_vel': 12,
    'dof_action': 12 * obs_mem_len,
    # 'targets': 12
})


class A1FlatCfg(LeggedRobotCfg):
    class customize:
        add_toe_force = False
        collect_errors = False
        pd_all_envs = None
        debugger_mode = 'none'
        debugger_sequence_len = 500
        state_sequence_len = 50
        observation_states = list(observation_states_size.keys())
        observation_states_size = observation_states_size
        use_state_mem = True  # the adding memory way by wangjing
        obs_mem_len = obs_mem_len   # 0.15*4 = 0.6
        obs_mem_skip = 3  # 3*0.05 = 0.15
        # the on the plate task
        add_plate = False
        relative_action = var_relative_action
        control_mode = var_control_mode
        runner_class = 'my'

    class init_state(LeggedRobotCfg.init_state):
        pos = var_init_pos  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        # default_joint_angles = {}
        # for leg in legs_name:
        #     default_joint_angles[f'{leg}_j0'] = np.radians(j0)
        #     default_joint_angles[f'{leg}_j1'] = np.radians(j1)
        #     default_joint_angles[f'{leg}_j2'] = np.radians(j2)
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'fl_j0': 0.1,  # [rad]
            'rl_j0': 0.1,  # [rad]
            'fr_j0': -0.1,  # [rad]
            'rr_j0': -0.1,  # [rad]

            'fl_j1': 0.8,  # [rad]
            'rl_j1': 1.,  # [rad]
            'fr_j1': 0.8,  # [rad]
            'rr_j1': 1.,  # [rad]

            'fl_j2': -1.5,  # [rad]
            'rl_j2': -1.5,  # [rad]
            'fr_j2': -1.5,  # [rad]
            'rr_j2': -1.5,  # [rad]
        }

    class env(LeggedRobotCfg.env):
        num_observations = sum(list(observation_states_size.values()))
        episode_length_s = 50

    class sim(LeggedRobotCfg.sim):
        dt = var_dt
        gravity = [0., 0., -9.81]  # [m/s^2]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0, 0]  # min max [rad/s]
            heading = [0, 0]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {
            'fr_j0': fore_j0_p, 'fr_j1': fore_j1_p, 'fr_j2': fore_j2_p,
            'fl_j0': fore_j0_p, 'fl_j1': fore_j1_p, 'fl_j2': fore_j2_p,

            'rr_j0': hind_j0_p, 'rr_j1': hind_j1_p, 'rr_j2': hind_j2_p,
            'rl_j0': hind_j0_p, 'rl_j1': hind_j1_p, 'rl_j2': hind_j2_p
        }

        for k, v in stiffness.items():
            stiffness[k] += 0

        damping = {
            'fr_j0': fore_j0_d, 'fr_j1': fore_j1_d, 'fr_j2': fore_j2_d,
            'fl_j0': fore_j0_d, 'fl_j1': fore_j1_d, 'fl_j2': fore_j2_d,

            'rr_j0': hind_j0_d, 'rr_j1': hind_j1_d, 'rr_j2': hind_j2_d,
            'rl_j0': hind_j0_d, 'rl_j1': hind_j1_d, 'rl_j2': hind_j2_d
        }
        for k, v in damping.items():
            damping[k] *= 1

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = var_action_scale
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = var_decimation

    class asset(LeggedRobotCfg.asset):
        file = os.path.join(current_root, 'meshes/a1/urdf/a1.urdf')
        name = "a1"
        foot_name = "_l3"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["_l0", "_l1", "base", "trunk"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True  # for the a1 robot, this should be True!!
        disable_gravity = False
        fix_base_link = var_fix_base_link

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 10 if var_relative_action else 100.

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 0.3]
        randomize_base_mass = False
        added_mass_ratio_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.15

        randomize_dof_friction = False
        dof_friction_range = [0, 1]

        randomize_pd = False
        stiffness = {
            '0': [3.0, 3.0], '1': [1., 2.], '2': [1., 2.],
        }

        damping = {
            '0': [0.001, 0.01], '1': [0.001, 0.01], '2': [0.001, 0.01],
        }


class A1FlatCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        observation_states = list(observation_states_size.keys())
        observation_states_size = dict(observation_states_size)
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'run1'
        experiment_name = 'flat_a1'
        save_interval = 30
        max_iterations = 4000

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from real_deployment.base_legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

import numpy as np
import os, sys

current_root = os.path.dirname(os.path.dirname(__file__))
# j0_default_ang = np.radians()
j0 = 4  # from 90 to 0
j1 = 15  # from 45 to -90
j2 = -15  # from -45 to 30

# j0 = 0
# j1 = 0
# j2 = 0

var_init_pos = [0.0, 0.0, 1.1]
var_fix_base_link = True
var_action_scale = 0.0
var_decimation = 100
var_dt = 0.001
var_control_mode = 'pos'

"""
hang_test
walk_test
"""

# ===== stiffness ====== #
# fore legs
fore_j0_p = 2.0
fore_j1_p = 0.4
fore_j2_p = 0.5

# hind legs
hind_j0_p = 2.0
hind_j1_p = 0.5
hind_j2_p = 0.8

# ===== damping ======= #
# fore legs
fore_j0_d = 0.01
fore_j1_d = 0.01
fore_j2_d = 0.01

# hind legs
hind_j0_d = 0.01
hind_j1_d = 0.01
hind_j2_d = 0.01


class WavegoFlatCfg(LeggedRobotCfg):
    class customize:
        add_toe_force = True
        collect_errors = False
        pd_all_envs = None
        debugger_mode = 'none'
        debugger_sequence_len = 500
        observation_states = [   # the order matters
            'row_pitch',
            'angular_v',
            'top_commands',
            'dof_pos',
            'dof_vel',
            'dof_action',
            'projected_gravity'
        ]

    class init_state(LeggedRobotCfg.init_state):
        pos = var_init_pos  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'fr_j0': np.radians(j0),  # [rad]
            'fr_j1': np.radians(j1),
            'fr_j2': np.radians(j2),

            'fl_j0': np.radians(j0),
            'fl_j1': np.radians(j1),
            'fl_j2': np.radians(j2),

            'rr_j0': np.radians(j0),
            'rr_j1': np.radians(j1),
            'rr_j2': np.radians(j2),

            'rl_j0': np.radians(j0),
            'rl_j1': np.radians(j1),
            'rl_j2': np.radians(j2),
        }

    class env(LeggedRobotCfg.env):
        num_observations = 42
        episode_length_s = 50

    class sim(LeggedRobotCfg.sim):
        dt = var_dt
        gravity = [0., 0., -9.81]  # [m/s^2]

    # class normalization(LeggedRobotCfg.normalization):
    #     clip_actions = 10

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
            lin_vel_x = [0.4, 0.4]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0, 0]  # min max [rad/s]
            heading = [0, 0]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {
            'fr_j0': fore_j0_p,
            'fr_j1': fore_j1_p,
            'fr_j2': fore_j2_p,

            'fl_j0': fore_j0_p,
            'fl_j1': fore_j1_p,
            'fl_j2': fore_j2_p,

            'rr_j0': hind_j0_p,
            'rr_j1': hind_j1_p,
            'rr_j2': hind_j2_p,

            'rl_j0': hind_j0_p,
            'rl_j1': hind_j1_p,
            'rl_j2': hind_j2_p
        }

        for k, v in stiffness.items():
            stiffness[k] += 10

        damping = {
            'fr_j0': fore_j0_d,
            'fr_j1': fore_j1_d,
            'fr_j2': fore_j2_d,

            'fl_j0': fore_j0_d,
            'fl_j1': fore_j1_d,
            'fl_j2': fore_j2_d,

            'rr_j0': hind_j0_d,
            'rr_j1': hind_j1_d,
            'rr_j2': hind_j2_d,

            'rl_j0': hind_j0_d,
            'rl_j1': hind_j1_d,
            'rl_j2': hind_j2_d
        }
        for k, v in damping.items():
            damping[k] *= 1

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = var_action_scale
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = var_decimation
        control_mode = var_control_mode

    class asset(LeggedRobotCfg.asset):
        file = os.path.join(current_root, 'meshes/wavego_v1/wavego_fix_j0.urdf')
        name = "wavego"
        foot_name = "_l3"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        disable_gravity = False
        fix_base_link = var_fix_base_link

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.1
        tracking_sigma = 0.05  # tracking reward = exp(-error^2/sigma)

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5

            torques = -0.000025
            dof_pos_limits = -10.0
            feet_air_time = 1.0

            termination = -0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.
            energy = -0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0, 2]
        randomize_base_mass = False
        added_mass_ratio_range = [-0.2, 0.2]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.
        randomize_dof_friction = False
        dof_friction_range = [0, 2]
        randomize_pd = False
        dof_stiffness_range = [10, 80]


class WavegoFlatCfgPPO(LeggedRobotCfgPPO):
    seed = 10

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'run1'
        experiment_name = 'flat_wavego'
        save_interval = 20
        max_iterations = 2000

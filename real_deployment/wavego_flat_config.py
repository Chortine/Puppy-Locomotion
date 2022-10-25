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
j0 = 0  # from 90 to 0
j1 = 0  # from 45 to -90
j2 = -35  # from -45 to 30

# j0 = 0
# j1 = 0
# j2 = 0

class WavegoFlatCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.1]  # x,y,z [m]
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
        dt = 0.005
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
            'fr_j0': 2.0,
            'fr_j1': 0.4,
            'fr_j2': 0.5,

            'fl_j0': 2.0,
            'fl_j1': 0.4,
            'fl_j2': 0.5,

            'rr_j0': 2.0,
            'rr_j1': 0.5,
            'rr_j2': 0.8,

            'rl_j0': 2.0,
            'rl_j1': 0.5,
            'rl_j2': 0.8
        }

        damping = {
            'fr_j0': 0.01,
            'fr_j1': 0.01,
            'fr_j2': 0.01,

            'fl_j0': 0.01,
            'fl_j1': 0.01,
            'fl_j2': 0.01,

            'rr_j0': 0.01,
            'rr_j1': 0.01,
            'rr_j2': 0.01,

            'rl_j0': 0.01,
            'rl_j1': 0.01,
            'rl_j2': 0.01
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # action_scale = 0.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 8

    class asset(LeggedRobotCfg.asset):
        file = os.path.join(current_root, 'meshes/wavego_v1/wavego_fix_j0.urdf')
        name = "wavego"
        foot_name = "_l3"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        disable_gravity = False
        fix_base_link = True

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
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'run1'
        experiment_name = 'flat_wavego'
        save_interval = 20
        max_iterations = 2000
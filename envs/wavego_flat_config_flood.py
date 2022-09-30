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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class WavegoFlatCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,  # [rad]
            'RL_hip_joint': 0.0,  # [rad]
            'FR_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': 0.0,  # [rad]

            'FL_thigh_joint': -0.2,  # [rad]
            'RL_thigh_joint': -0.5,  # [rad]
            'FR_thigh_joint': -0.2,  # [rad]
            'RR_thigh_joint': -0.5,  # [rad]

            'FL_calf_joint': 0.2,  # [rad]
            'RL_calf_joint': 0.0,  # [rad]
            'FR_calf_joint': 0.2,  # [rad]
            'RR_calf_joint': 0.0,  # [rad]
        }

    class env(LeggedRobotCfg.env):
        num_observations = 48
        episode_length_s = 50

    class sim(LeggedRobotCfg.sim):
        dt = 0.02

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        all_stiffness = 20
        all_damping = 0.5
        all_friction = 0.0
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1

    class asset(LeggedRobotCfg.asset):
        file = '/home/tianchu/Documents/code_qy/puppy-gym/meshes/wavego/urdf/wavego.urdf'
        name = "wavego"
        foot_name = "foot"
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["base", "thigh"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        angular_damping = 0.
        linear_damping = 0.
        fix_base_link = False
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.000025
            dof_pos_limits = -10.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0, 2]
        randomize_base_mass = True
        added_mass_ratio_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_dof_friction = True
        dof_friction_range = [0, 2]
        randomize_pd = False
        dof_stiffness_range = [10, 80]

    class commands:
        curriculum = True
        max_curriculum = 2.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]
            # lin_vel_x = [0.0, 0.0] # min max [m/s]
            # lin_vel_y = [1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [0, 0]    # min max [rad/s]
            # heading = [0, 0]


class WavegoFlatCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'run1'
        experiment_name = 'flat_wavego'
        save_interval = 500

        max_iterations = 2000




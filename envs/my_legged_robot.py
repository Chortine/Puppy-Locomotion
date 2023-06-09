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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
# from real_deployment.base_legged_robot_config import LeggedRobotCfg
from envs.wavego_flat_config import WavegoFlatCfg


# from envs.utils.terrain_generation import *


class LeggedRobot(BaseTask):
    def __init__(self, cfg: WavegoFlatCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)

        if self.cfg.customize.use_env_factors:
            self._refresh_env_factors_buffer(mode='init')

        self.pd_all_envs = self.cfg.customize.pd_all_envs
        # randomize pd for all envs
        if self.cfg.domain_rand.randomize_pd:
            self.pd_all_envs = torch.zeros(self.cfg.env.num_envs, 6, dtype=torch.float, device=sim_device,
                                           requires_grad=False)
            for env in range(self.cfg.env.num_envs):
                for i in range(3):
                    p_range = self.cfg.domain_rand.stiffness[str(i)]
                    self.pd_all_envs[env][i] = p_range[0] + (p_range[1] - p_range[0]) * np.random.uniform()
                for i in range(3):
                    d_range = self.cfg.domain_rand.damping[str(i)]
                    self.pd_all_envs[env][i + 3] = d_range[0] + (d_range[1] - d_range[0]) * np.random.uniform()

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        if self.cfg.customize.add_toe_force:
            self.refresh_forces_at_toe()

        if self.cfg.customize.relative_action:
            scaled_actions = self.cfg.control.action_scale * self.actions
            self.targets = scaled_actions + self.last_targets

        else:
            self.targets = self.cfg.control.action_scale * self.actions + self.default_dof_pos
        # clip the targets
        self.targets = torch.clip(self.targets, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        self.last_targets = torch.clone(self.targets)

        for _ in range(self.cfg.control.decimation):
            if self.cfg.customize.add_toe_force:
                self.apply_forces_at_toe()

            if self.cfg.customize.control_mode == 'pos':
                self.torques = self._compute_torques_from_targets(self.targets).view(self.torques.shape)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.targets))
            else:
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # resample the env factor
        if self.cfg.customize.use_env_factors:
            self._refresh_env_factors_buffer(mode='reset', env_ids=env_ids)

        # it's important; otherwise set_actor_rigid_body_properties will effect the root state
        self.gym.simulate(self.sim)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_targets[env_ids] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                self.friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                         device=self.device)
                self.friction_coeffs = self.friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                props[s].rolling_friction = self.friction_coeffs[env_id]
                props[s].torsion_friction = self.friction_coeffs[env_id]

        # ============== randomize restitution ============== #
        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                rest_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                self.rest_buckets = torch_rand_float(rest_range[0], rest_range[1], (num_buckets, 1),
                                                         device=self.device)
                self.restitution = self.rest_buckets[bucket_ids]

            for s in range(len(props)):
                body_name = self.body_names[s]
                # props[s].contact_offset = 0.005
                if '_l3' in body_name or '_l2' in body_name:
                    # props[s].compliance = 1000.0
                    props[s].restitution = self.restitution[env_id]
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass

        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                # prepare friction randomization
                mass_range = self.cfg.domain_rand.added_mass_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                self.mass_buckets = torch_rand_float(mass_range[0], mass_range[1], (num_buckets, 1),
                                                     device=self.device)
                self.payloads = self.mass_buckets[bucket_ids]
                self.default_base_mass = props[0].mass

            props[0].mass += self.payloads[env_id]

        # if self.cfg.domain_rand.randomize_base_mass:
        #     rng = self.cfg.domain_rand.added_mass_range
        #     props[0].mass += np.random.uniform(rng[0], rng[1])

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        props = self.get_modified_dof_props(props, env_id)
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.soft_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits. only used to compute the reward
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.soft_dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.soft_dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        if self.cfg.domain_rand.randomize_dof_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.dof_friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device=self.device)
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props['friction'][s] = self.friction_coeffs[env_id]

        # if self.cfg.domain_rand.randomize_pd:
        #     if env_id == 0:
        #         # prepare pd randomization
        #         stiffness_range = self.cfg.domain_rand.dof_stiffness_range
        #         num_buckets = 64
        #         bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
        #         stiffness_buckets = torch_rand_float(stiffness_range[0], stiffness_range[1], (num_buckets, 1),
        #                                              device='cpu')
        #         self.stiffness_coeffs = stiffness_buckets[bucket_ids]
        # for s in range(len(props)):
        #     if s % 3 == 0:
        #         props['stiffness'][s] = 2
        #     elif s % 3 == 1:
        #         props['stiffness'][s] = 0.4
        #     else:
        #         props['stiffness'][s] = 0.5
        #     props['damping'][s] = 0.01
        #     props['effort'][s] = 1.

        return props

    def refresh_forces_at_toe(self):
        """
        For each rl steps, refresh the external force disturbance
        """
        x_range = 5.0  # N
        y_range = 5.0  # N
        z_range = 10  # N
        # the final force tensor : (env_nums, 3)
        # same force for all envs, but different forces for different legs
        fr_xyz = np.random.uniform(low=[-x_range, -y_range, -z_range], high=[x_range, y_range, z_range])
        fl_xyz = np.random.uniform(low=[-x_range, -y_range, -z_range], high=[x_range, y_range, z_range])
        rl_xyz = np.random.uniform(low=[-x_range, -y_range, -z_range], high=[x_range, y_range, z_range])
        rr_xyz = np.random.uniform(low=[-x_range, -y_range, -z_range], high=[x_range, y_range, z_range])

        # random_forces_x = torch.rand(self.num_envs) * 2 * x_range - x_range
        # random_forces_y = torch.rand(self.num_envs) * 2 * y_range - y_range
        # random_forces_z = torch.rand(self.num_envs) * 2 * z_range - z_range
        # change force in self.external_forces
        tensor_fr_xyz = torch.Tensor(fr_xyz).repeat(self.num_envs, 1)
        tensor_fl_xyz = torch.Tensor(fl_xyz).repeat(self.num_envs, 1)
        tensor_rl_xyz = torch.Tensor(rl_xyz).repeat(self.num_envs, 1)
        tensor_rr_xyz = torch.Tensor(rr_xyz).repeat(self.num_envs, 1)

        for i, name in enumerate(self.body_names):
            # change only one leg is enough
            if 'fr_l3' in name:
                self.external_forces[..., i, :] = tensor_fr_xyz
            if 'fl_l3' in name:
                self.external_forces[..., i, :] = tensor_fl_xyz
            if 'rl_l3' in name:
                self.external_forces[..., i, :] = tensor_rl_xyz
            if 'rr_l3' in name:
                self.external_forces[..., i, :] = tensor_rr_xyz

        return

    def apply_forces_at_toe(self):
        # self.toe_forces = torch.rand(self.toe_forces)
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim,
                                                       forceTensor=gymtorch.unwrap_tensor(self.external_forces),
                                                       posTensor=gymtorch.unwrap_tensor(self.external_forces_pos),
                                                       space=gymapi.LOCAL_SPACE)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            if self.cfg.customize.tilted_plane:
                self.commands[env_ids, 3] = self.init_heading[env_ids]
            else:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                             self.command_ranges["heading"][1], (len(env_ids), 1),
                                                             device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

        # set small commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        # actions_scaled[0, 4] = -0.35
        # actions_scaled[0, 5] = 0.8
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (
                    actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        rigid_contact_info = self.contact_forces
        # self.gym.ENV_SPACE
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_torques_from_targets(self, targets):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        # actions_scaled[0, 4] = -0.35
        # actions_scaled[0, 5] = 0.8
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (
                    targets - self.dof_pos) - self.d_gains * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        rigid_contact_info = self.contact_forces
        # self.gym.ENV_SPACE
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(-0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.customize.tilted_plane:
                z_drift = - self.env_origins[env_ids][..., 0] * np.tan(
                    np.deg2rad(self.cfg.customize.plane_tilted_angle))
                self.root_states[env_ids, 2] += z_drift
                # reset_root_states_rot
                rolls = torch.zeros(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False)
                pitchs = torch.zeros(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False)
                yaws = torch.squeeze(torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device))
                quats = quat_from_euler_xyz(rolls, pitchs, yaws)
                self.root_states[env_ids, 3:7] = quats
                forward = quat_apply(quats, self.forward_vec[env_ids])
                # change the init heading
                self.init_heading[env_ids] = torch.atan2(forward[:, 1], forward[:, 0])

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
                                                    device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self, cfg):
        # print("==== _get_noise_scale_vec ====")
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        link_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.link_state = gymtorch.wrap_tensor(link_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_errors = torch.zeros_like(self.dof_pos)
        self.mean_dof_errors = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        self.link_pos = self.link_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]

        self.root_states = self.root_states[:self.num_envs, ...]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)

        forward = quat_apply(self.base_quat, self.forward_vec)
        self.heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.init_heading = self.heading
        # inclination: (num_envs, 2), which is in x and y directions
        self.inclination = torch.stack(
            [torch.cos(self.heading) * np.deg2rad(self.cfg.customize.plane_tilted_angle),
             -torch.sin(self.heading) * np.deg2rad(self.cfg.customize.plane_tilted_angle)],
            dim=1)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.last_targets = self.default_dof_pos.repeat(self.num_envs, 1)
        self._init_pd_gains_buffer()

    def _refresh_env_factors_buffer(self, mode='init', env_ids=None):
        """
        This will be used to turn on the randomization switch.
        So it should be called before _create_envs()
        :param mode:
        :param env_ids:
        :return:
        """
        assert mode in ['init', 'reset']
        if mode == 'init':
            # init the env factor bucket
            if 'payload' in self.cfg.customize.env_factors:
                self.cfg.domain_rand.randomize_base_mass = True
            if 'dof_stiffness' in self.cfg.customize.env_factors:
                # for each env, should be a 12-length vec
                self.cfg.domain_rand.randomize_pd = True
            if 'dof_damping' in self.cfg.customize.env_factors:
                self.cfg.domain_rand.randomize_pd = True
            if 'terrain_friction' in self.cfg.customize.env_factors:
                self.cfg.domain_rand.randomize_friction = True
            if 'restitution' in self.cfg.customize.env_factors:
                self.cfg.domain_rand.randomize_restitution = True

        elif mode == 'reset':
            assert env_ids is not None
            # reset the env factor for certain env_ids
            if 'payload' in self.cfg.customize.env_factors:
                bucket_ids = torch.randint(0, 64, (len(env_ids), 1))
                self.payloads[env_ids] = self.mass_buckets[bucket_ids]

            if 'terrain_friction' in self.cfg.customize.env_factors:
                bucket_ids = torch.randint(0, 64, (len(env_ids), 1))
                self.friction_coeffs[env_ids] = self.friction_buckets[bucket_ids]

            if 'restitution' in self.cfg.customize.env_factors:
                bucket_ids = torch.randint(0, 64, (len(env_ids), 1))
                self.restitution[env_ids] = self.rest_buckets[bucket_ids]

            if 'dof_stiffness' in self.cfg.customize.env_factors or 'dof_damping' in self.cfg.customize.env_factors:
                for env_id in env_ids:
                    for i in range(3):
                        p_range = self.cfg.domain_rand.stiffness[str(i)]
                        self.pd_all_envs[env_id][i] = p_range[0] + (p_range[1] - p_range[0]) * np.random.uniform()
                    for i in range(3):
                        d_range = self.cfg.domain_rand.damping[str(i)]
                        self.pd_all_envs[env_id][i + 3] = d_range[0] + (d_range[1] - d_range[0]) * np.random.uniform()
                    for dof_id, dof_name in enumerate(self.dof_names):
                        if '_j0' in dof_name:
                            self.p_gains[env_id][dof_id] = self.pd_all_envs[env_id][0]
                            self.d_gains[env_id][dof_id] = self.pd_all_envs[env_id][3]
                        elif '_j1' in dof_name:
                            self.p_gains[env_id][dof_id] = self.pd_all_envs[env_id][1]
                            self.d_gains[env_id][dof_id] = self.pd_all_envs[env_id][4]
                        elif '_j2' in dof_name:
                            self.p_gains[env_id][dof_id] = self.pd_all_envs[env_id][2]
                            self.d_gains[env_id][dof_id] = self.pd_all_envs[env_id][5]
            # ========= modify properties according to the modified buffer
            for env_id in env_ids:
                # 1. modify rigid body properties
                body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id])
                body_props[0].mass = self.default_base_mass + self.payloads[env_id]
                self.gym.set_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id], body_props,
                                                         recomputeInertia=True)
                # 2. modify rigid shape properties
                shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id])
                for prop, body_name in zip(shape_props, self.body_names):  # for every rigid shape in this env
                    prop.friction = self.friction_coeffs[env_id]
                    prop.rolling_friction = self.friction_coeffs[env_id]
                    prop.torsion_friction = self.friction_coeffs[env_id]
                    # props[s].contact_offset = 0.005
                    if '_l3' in body_name or '_l2' in body_name:
                        # props[s].compliance = 1000.0
                        prop.restitution = self.restitution[env_id]

                self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id], shape_props)
                # 3. modify dof properties
                dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], self.actor_handles[env_id])
                dof_props = self.get_modified_dof_props(dof_props, env_id)
                self.gym.set_actor_dof_properties(self.envs[env_id], self.actor_handles[env_id], dof_props)

    def _init_pd_gains_buffer(self):
        for env_id in range(self.num_envs):
            if self.pd_all_envs is not None:
                for dof_id, dof_name in enumerate(self.dof_names):
                    if '_j0' in dof_name:
                        self.p_gains[env_id][dof_id] = self.pd_all_envs[env_id][0]
                        self.d_gains[env_id][dof_id] = self.pd_all_envs[env_id][3]
                    elif '_j1' in dof_name:
                        self.p_gains[env_id][dof_id] = self.pd_all_envs[env_id][1]
                        self.d_gains[env_id][dof_id] = self.pd_all_envs[env_id][4]
                    elif '_j2' in dof_name:
                        self.p_gains[env_id][dof_id] = self.pd_all_envs[env_id][2]
                        self.d_gains[env_id][dof_id] = self.pd_all_envs[env_id][5]
            else:
                for dof_id, dof_name in enumerate(self.dof_names):
                    self.p_gains[env_id][dof_id] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[env_id][dof_id] = self.cfg.control.damping[dof_name]

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        if self.cfg.customize.tilted_plane:
            tilted_angle = np.deg2rad(self.cfg.customize.plane_tilted_angle)
            normal = np.asarray([np.sin(tilted_angle), 0.0, np.cos(tilted_angle)])
            normal /= np.linalg.norm(normal)
        else:
            normal = [0.0, 0.0, 1.0]
        plane_params.normal = gymapi.Vec3(*normal)
        plane_params.static_friction = 0.0
        plane_params.dynamic_friction = 0.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_ground_plane2(self):
        # Terrain specifications
        terrain_width = 50  # terrain width [m]
        terrain_length = terrain_width  # terrain length [m]
        self.terrain_side_length = terrain_width
        if terrain_length != terrain_width:
            print("!!!   terrain width != terrain height, PLEASE FIX   !!!")
        # KEEP TERRAIN WIDTH AND LENGTH EQUAL!!! - check_spawn_slope will not work if the are not.
        horizontal_scale = 0.05  # 0.025
        # resolution per meter
        vertical_scale = 0.005  # vertical resolution [m]
        self.heightfield = np.zeros((int(terrain_width / horizontal_scale), int(terrain_length / horizontal_scale)),
                                    dtype=np.int16)

        def new_sub_terrain():
            return SubTerrain1(width=terrain_width, length=terrain_length, horizontal_scale=horizontal_scale,
                               vertical_scale=vertical_scale)

        terrain = gaussian_terrain(new_sub_terrain(), 0.5, 0.0)
        terrain = gaussian_terrain(terrain, 15, 5)
        # rock_heigtfield, self.rock_positions = add_rocks_terrain(terrain=terrain, rock_height=(0.05, 0.1))
        # self.heightfield[0:int(terrain_width / horizontal_scale), :] = rock_heigtfield.height_field_raw
        # vertices, triangles = convert_heightfield_to_trimesh1(self.heightfield, horizontal_scale=horizontal_scale,
        #                                                       vertical_scale=vertical_scale, slope_threshold=None)
        # Decimate mesh and reduce number of vertices
        vertices, triangles = convert_heightfield_to_trimesh1(terrain.height_field_raw,
                                                              horizontal_scale=horizontal_scale,
                                                              vertical_scale=vertical_scale, slope_threshold=None)
        vertices, triangles = polygon_reduction(vertices, triangles, target_vertices=20000)  # default is 200000

        self.tensor_map = torch.tensor(self.heightfield, device=self.device)
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        tm_params = gymapi.TriangleMeshParams()
        tm_params.static_friction = 1.0
        tm_params.dynamic_friction = 1.0
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        # If the gound plane should be shifted:
        self.shift = -5
        tm_params.transform.p.x = self.shift
        tm_params.transform.p.y = self.shift
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        if self.cfg.customize.add_plate:
            plate_root = '/home/tianchu/Documents/code_qy/puppy-gym/meshes/wavego_v1'
            plate_file = 'plate.urdf'
            plate_asset = self.gym.load_asset(self.sim, plate_root, plate_file, asset_options)
            self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset) + self.gym.get_asset_rigid_body_count(
                plate_asset)
        else:
            self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.body_names = body_names
        # self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        # self.cfg.init_state.pos[2] = -1.0
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # self.base_init_state *= 0.0

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        if self.cfg.customize.add_plate:
            # add env handle
            start_pose = gymapi.Transform()
            for i in range(self.num_envs):
                pos = self.env_origins[i].clone()
                # pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

                # get env handle
                env_handle = self.gym.get_env(self.sim, i)
                # add plate as an actor
                self.gym.create_actor(env_handle, plate_asset, start_pose, 'plate', i,
                                      self.cfg.asset.self_collisions, 0)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def get_modified_dof_props(self, props, env_id):
        """
        modified the gym dof props according to the env_cfg. Default it's retrived from urdf.
        :param props:
        :param env_id:
        :return:
        """
        for dof_id, dof_name in enumerate(self.dof_names):
            if self.cfg.customize.control_mode == 'pos':
                props['driveMode'][dof_id] = gymapi.DOF_MODE_POS  # 1
                if self.pd_all_envs is not None:
                    if '_j0' in dof_name:
                        props['stiffness'][dof_id] = self.pd_all_envs[env_id][0]
                        props['damping'][dof_id] = self.pd_all_envs[env_id][3]
                    elif '_j1' in dof_name:
                        props['stiffness'][dof_id] = self.pd_all_envs[env_id][1]
                        props['damping'][dof_id] = self.pd_all_envs[env_id][4]
                    elif '_j2' in dof_name:
                        props['stiffness'][dof_id] = self.pd_all_envs[env_id][2]
                        props['damping'][dof_id] = self.pd_all_envs[env_id][5]
                else:
                    props['stiffness'][dof_id] = self.cfg.control.stiffness[dof_name]
                    props['damping'][dof_id] = self.cfg.control.damping[dof_name]
            elif self.cfg.customize.control_mode == 'torque':
                props['driveMode'][dof_id] = gymapi.DOF_MODE_EFFORT  # 3
                # IMPORTANT!!! In the dof_mode_effort mode the pd should be zero.
                props['stiffness'][dof_id] = 0.0
                props['damping'][dof_id] = 0.0
        return props

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.soft_dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.soft_dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        # return -lin_vel_error
        # lin_vel = torch.sum(self.base_lin_vel[:, :2], dim=1)
        # return lin_vel

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact,
                                dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
                torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_toe_height(self):
        j1 = torch.index_select(self.dof_pos, 1, torch.tensor([1, 4, 7, 10]).to("cuda"))
        j2 = torch.index_select(self.dof_pos, 1, torch.tensor([2, 5, 8, 11]).to("cuda"))
        alpha = j1 + torch.deg2rad(torch.tensor([45])).to("cuda")
        gamma = j2 + torch.deg2rad(torch.tensor([120])).to("cuda")
        theta = torch.deg2rad(torch.tensor([180])).to("cuda") - alpha - gamma
        toe_from_body = torch.tensor([40.0]).to("cuda") * torch.cos(alpha) + torch.tensor(70.).to("cuda") * torch.cos(theta)

        # slow
        reward = torch.abs(toe_from_body[:, 0] - toe_from_body[:, 1]) \
                 + torch.abs(toe_from_body[:, 2] - toe_from_body[:, 3]) \
                 + torch.abs(toe_from_body[:, 0] - toe_from_body[:, 2]) \
                 + torch.abs(toe_from_body[:, 1] - toe_from_body[:, 3]) \
                 - torch.abs(toe_from_body[:, 0] - toe_from_body[:, 3]) \
                 - torch.abs(toe_from_body[:, 1] - toe_from_body[:, 2])

        return reward
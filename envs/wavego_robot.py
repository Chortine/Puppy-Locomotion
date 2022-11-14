# from legged_gym.envs import LeggedRobot
import time

from envs.base.customized_legged_robot import LeggedRobot
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
import torch
import numpy as np
from collections import deque
import os, sys
import pickle
from real_deployment.transition_debugger import TransitionDebugger
import matplotlib.pyplot as plt

current_folder = os.path.dirname(__file__)


class WavegoRobot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # self.action = None
        if self.cfg.customize.debugger_mode != 'none':
            self.debugger = TransitionDebugger(mode=self.cfg.customize.debugger_mode,
                                               sequence_len=self.cfg.customize.debugger_sequence_len,
                                               transition_path='/home/jingjing/PycharmProjects/puppy-bot/logs')
        self.num_envs = self.cfg.customize.num_envs
        self.obs_mem_len = self.cfg.customize.obs_mem_len  # 4
        self.obs_mem_skip = self.cfg.customize.obs_mem_skip  # 3
        mem_max_len = self.obs_mem_skip * self.obs_mem_len
        self.mem = {"s_roll": deque(maxlen=mem_max_len),
                    "c_roll": deque(maxlen=mem_max_len),
                    "s_pitch": deque(maxlen=mem_max_len),
                    "c_pitch": deque(maxlen=mem_max_len),
                    "base_ang_vel": deque(maxlen=mem_max_len),
                    "actions": deque(maxlen=mem_max_len)}

        for i in range(mem_max_len):
            self.mem["s_roll"].append(torch.zeros(self.num_envs, 1).cuda())
            self.mem["c_roll"].append(torch.ones(self.num_envs, 1).cuda())
            self.mem["s_pitch"].append(torch.zeros(self.num_envs, 1).cuda())
            self.mem["c_pitch"].append(torch.ones(self.num_envs, 1).cuda())
            self.mem["base_ang_vel"].append(torch.zeros(self.num_envs, 3).cuda())
            self.mem["actions"].append(torch.zeros(self.num_envs, 12).cuda())

    def _init_buffers(self):
        super()._init_buffers()
        self.dof_vel_from_deviation = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_pos[:] = self.default_dof_pos[:]

    def step(self, actions):
        super().step(actions)
        # add Nan exception
        nan_array = torch.isnan(self.obs_buf)
        nan_idx = nan_array.nonzero().squeeze(-1)
        if len(nan_idx) > 0:
            self.reset_buf[nan_idx[:, 0]] = 1
            self.rew_buf[nan_idx[:, 0]] = 0.0
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self.reset_idx(env_ids)
            self.obs_buf[nan_array] = 0.0
            tmp = nan_idx.detach().cpu().numpy()
            tmp = np.unique(tmp[:, 0])
            print(f'!!! has nan on env {tmp}')

            nan_array = torch.isnan(self.obs_buf)
            nan_idx = nan_array.nonzero().squeeze(-1)
            if len(nan_idx) > 0:
                tmp = nan_idx.detach().cpu().numpy()
                tmp = np.unique(tmp[:, 0])
                print(f'!!! Still has nan on env {tmp}')
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _obs_mem(self, s_roll, c_roll, s_pitch, c_pitch):
        self.mem["s_roll"].append(s_roll)
        self.mem["c_roll"].append(c_roll)
        self.mem["s_pitch"].append(s_pitch)
        self.mem["c_pitch"].append(c_pitch)
        self.mem["base_ang_vel"].append(self.base_ang_vel * self.obs_scales.ang_vel)
        self.mem["actions"].append(self.actions)

        result = {}
        for key in self.mem.keys():
            result[key] = self.mem[key][-1]
            for i in range(self.obs_mem_len-1):
                result[key] = torch.concat((result[key], self.mem[key][-1 - self.obs_mem_skip*i - self.obs_mem_skip]), 1)
        return result

    def compute_observations(self):
        """ Computes observations
        """
        # my observations
        self.dof_vel_from_deviation = (self.dof_pos - self.last_dof_pos) / self.dt
        roll, pitch, _ = get_euler_xyz(self.base_quat)
        roll += 0.1 * torch.rand(self.cfg.customize.num_envs,).cuda()
        pitch += 0.1 * torch.rand(self.cfg.customize.num_envs,).cuda()
        s_roll, c_roll, s_pitch, c_pitch = torch.unsqueeze(torch.sin(roll), 1), \
                                           torch.unsqueeze(torch.cos(roll), 1), \
                                           torch.unsqueeze(torch.sin(pitch), 1), \
                                           torch.unsqueeze(torch.cos(pitch), 1)
        mem_obs = self._obs_mem(s_roll, c_roll, s_pitch, c_pitch)
        self.obs_buf = torch.cat((
            mem_obs["base_ang_vel"],  # 3
            mem_obs["s_roll"],
            mem_obs["c_roll"],
            mem_obs["s_pitch"],
            mem_obs["c_pitch"],
            self.commands[:, :3] * self.commands_scale,
            mem_obs["actions"] + 0.05*torch.rand(self.cfg.customize.num_envs, 12*self.cfg.customize.obs_mem_len).cuda()
        ), dim=-1)

        # self.obs_buf = torch.cat((
        #     (self.base_ang_vel + 0.3 * torch.rand(10, 3).cuda()) * self.obs_scales.ang_vel,  # 3
        #     s_roll,
        #     c_roll,
        #     s_pitch,
        #     c_pitch,
        #     self.commands[:, :3] * self.commands_scale,
        #     (self.dof_pos - self.default_dof_pos + 0.05*torch.rand(10, 12).cuda()) * self.obs_scales.dof_pos,
        #     (self.dof_vel_from_deviation + 0.05*torch.rand(10, 12).cuda()) * self.obs_scales.dof_vel,
        #     self.actions + 0.05*torch.rand(10, 12).cuda()
        # ), dim=-1)
        # print("obs_buf: ", self.obs_buf)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_dof_pos[:] = self.dof_pos[:]

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

        # collect errors for pd_search
        if self.cfg.customize.collect_errors:
            dof_pos_error = torch.abs(self.targets - self.dof_pos)
            self.dof_errors += dof_pos_error
            self.error_add_count += 1
            self.mean_dof_errors = torch.mean(self.dof_errors, dim=-1) / self.error_add_count
            # if self.debugger.replay_done:
            if self.common_step_counter == 100:
                # print only the first robot's error
                robot_0_dof_error = self.mean_dof_errors[0]
                robot_0_dof_error = robot_0_dof_error.detach().cpu().numpy()
                print(f'replay done; the first robot dof error is {robot_0_dof_error}')

        # collect debugger data
        if self.cfg.customize.debugger_mode != 'none':
            toe_pose = self.get_toe_pose()
            actions_scaled = np.squeeze(self.actions.detach().cpu().numpy()) * self.cfg.control.action_scale
            # if self.action is not None:
            if self.cfg.customize.debugger_mode == 'collect':
                self.debugger.step(obs=self.obs_buf.detach().cpu(), action=actions_scaled, toe_pose=toe_pose)
            elif self.cfg.customize.debugger_mode == 'replay':
                obs = self.debugger.step(obs=self.obs_buf.detach().cpu(), action=actions_scaled, toe_pose=toe_pose)
                obs = obs[0].repeat(self.num_envs, 1)
                self.obs_buf = torch.tensor(obs, device=self.sim_device)

        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def get_toe_pose(self):
        """
        This can be correct only when the base is fixed
        Get the wiggle_angle[] and toe xy on the wiggle plane.
        The order of pos l3 from sim is: fl, fr, rl, rr
        The output order of leg is: rr, fr, fl, rl
        The toe pose is to check if the forward kinematics in pipeline has problem
        :return: toe_pose (wiggle_angle[degrees], l2_end_x_leg[mm], l2_end_y_leg[mm])
        """
        # alphas = np.deg2rad(0.0)
        alphas = {}
        link_name = self.body_names
        dof_pos = np.squeeze(self.dof_pos.detach().cpu().numpy()[0])
        dof_name_pos = {key: dof_pos[i] for i, key in enumerate(self.dof_names)}
        for leg in ['rr', 'fr', 'fl', 'rl']:
            alphas[leg] = dof_name_pos[f'{leg}_j0']
        link_pos = np.squeeze(self.link_pos.detach().cpu().numpy()[0])
        link_name_pos = {key: link_pos[i, :] for i, key in enumerate(link_name)}
        # compute the relative pos from the COM l3 to COM l0
        relative_toe_xyz_l0 = {}
        relative_toe_xyz_thigh_center = {}
        # drift from center of thigh to the COM l0, represented in l0 coordinate
        drift_from_thigh_center = {  # in meters
            'fr': [0.017, 0, -0.007],
            'fl': [0.017, 0, -0.007],
            'rl': [-0.017, 0, -0.007],
            'rr': [-0.017, 0, -0.007]
        }
        for leg in ['rr', 'fr', 'fl', 'rl']:
            # relative pose in world coordinate
            relative_toe_xyz_world = link_name_pos[f'{leg}_l3'] - link_name_pos[f'{leg}_l0']
            # transfer it in l0 coordinate:
            if leg in ['fr', 'rr']:
                z_in_l0 = relative_toe_xyz_world[1] * np.sin(alphas[leg]) + relative_toe_xyz_world[2] * np.cos(
                    alphas[leg])
                # the y is just for debugging. to see if it's a constant
                y_in_l0 = relative_toe_xyz_world[1] * np.cos(alphas[leg]) - relative_toe_xyz_world[2] * np.sin(
                    alphas[leg])
            elif leg in ['fl', 'rl']:
                z_in_l0 = relative_toe_xyz_world[1] * np.sin(-alphas[leg]) + relative_toe_xyz_world[2] * np.cos(
                    -alphas[leg])
                y_in_l0 = relative_toe_xyz_world[1] * np.cos(-alphas[leg]) - relative_toe_xyz_world[2] * np.sin(
                    -alphas[leg])

            relative_toe_xyz_l0[leg] = [relative_toe_xyz_world[0], y_in_l0, z_in_l0]
            relative_toe_xyz_thigh_center[leg] = relative_toe_xyz_l0[leg] - np.asarray(drift_from_thigh_center[leg])

        # print(f'relative_toe_xyz_l0_rr: {relative_toe_xyz_thigh_center["rr"][1]}')
        # transfer from sim coordinate to real leg coordinate
        relative_toe_pose_thigh_center_real_coord = {}
        relative_toe_pose_thigh_center_real_coord['rr'] = [alphas['rr'],
                                                           relative_toe_xyz_thigh_center['rr'][0],
                                                           -relative_toe_xyz_thigh_center['rr'][2]]
        relative_toe_pose_thigh_center_real_coord['fr'] = [alphas['fr'],
                                                           relative_toe_xyz_thigh_center['fr'][0],
                                                           -relative_toe_xyz_thigh_center['fr'][2]]
        relative_toe_pose_thigh_center_real_coord['fl'] = [alphas['fl'],
                                                           relative_toe_xyz_thigh_center['fl'][0],
                                                           -relative_toe_xyz_thigh_center['fl'][2]]
        relative_toe_pose_thigh_center_real_coord['rl'] = [alphas['rl'],
                                                           relative_toe_xyz_thigh_center['rl'][0],
                                                           -relative_toe_xyz_thigh_center['rl'][2]]
        for key, value in relative_toe_pose_thigh_center_real_coord.items():
            value[0] = np.rad2deg(value[0])
            value[1] *= 1000
            value[2] *= 1000
        toe_pos = [relative_toe_pose_thigh_center_real_coord['rr'],
                   relative_toe_pose_thigh_center_real_coord['fr'],
                   relative_toe_pose_thigh_center_real_coord['fl'],
                   relative_toe_pose_thigh_center_real_coord['rl']]
        toe_pos = np.concatenate(toe_pos)
        return toe_pos

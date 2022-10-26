# from legged_gym.envs import LeggedRobot
import time

from envs.base.customized_legged_robot import LeggedRobot
import torch
import numpy as np
from collections import deque
import os, sys
import pickle
from real_deployment.transition_debugger import TransitionDebugger


class WavegoRobot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.sim_device = sim_device
        # self.action = None
        self.debugger_mode = 'collect'
        if self.debugger_mode != 'none':
            self.debugger = TransitionDebugger(mode=self.debugger_mode, sequence_len=500,
                                               transition_path='/home/tianchu/Documents/code_qy/puppy-gym/envs/data')

    def _init_buffers(self):
        super()._init_buffers()
        self.dof_vel_from_deviation = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_pos[:] = self.default_dof_pos[:]

    def step(self, actions):
        super().step(actions)
        # self.action = np.squeeze(actions.detach().cpu().numpy())
        # add Nan exception
        nan_array = torch.isnan(self.obs_buf)
        nan_idx = nan_array.nonzero().squeeze(-1)
        if len(nan_idx) > 0:
            # tmp1 = self.obs_buf.detach().cpu().numpy()
            self.reset_buf[nan_idx[:, 0]] = 1
            self.rew_buf[nan_idx[:, 0]] = 0.0
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self.reset_idx(env_ids)
            self.obs_buf[nan_array] = 0.0
            # tmp2 = self.obs_buf.detach().cpu().numpy()
            # tmp3 = tmp2 - tmp1
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
        dof_pos = np.squeeze(self.dof_pos.detach().cpu().numpy())
        dof_name_pos = {key: dof_pos[i] for i, key in enumerate(self.dof_names)}
        for leg in ['rr', 'fr', 'fl', 'rl']:
            alphas[leg] = dof_name_pos[f'{leg}_j0']
        link_pos = np.squeeze(self.link_pos.detach().cpu().numpy())
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

    def compute_observations(self):
        """
        Computes observations
        """
        time.sleep(0.1)
        # my observations
        self.dof_vel_from_deviation = (self.dof_pos - self.last_dof_pos) / self.dt
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            # self.projected_gravity,  # 3
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel_from_deviation * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        toe_pose = self.get_toe_pose()
        if self.debugger_mode != 'none':
            actions_scaled = np.squeeze(self.actions.detach().cpu().numpy()) * self.cfg.control.action_scale
            # if self.action is not None:
            if self.debugger_mode == 'collect':
                self.debugger.step(obs=self.obs_buf.detach().cpu(), action=actions_scaled, toe_pose=toe_pose)
            elif self.debugger_mode == 'replay':
                obs = self.debugger.step(obs=self.obs_buf.detach().cpu(), action=actions_scaled, toe_pose=toe_pose)
                self.obs_buf = torch.tensor(obs, device=self.sim_device)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_dof_pos[:] = self.dof_pos[:]

    # def render(self):

    def _reward_energy(self):
        return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)

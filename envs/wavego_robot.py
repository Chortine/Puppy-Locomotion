# from legged_gym.envs import LeggedRobot
from envs.my_legged_robot import LeggedRobot
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse
import torch
import numpy as np
from collections import deque
import os, sys
import pickle
from real_deployment.transition_debugger import TransitionDebugger
import matplotlib.pyplot as plt
from isaacgym.torch_utils import *

current_folder = os.path.dirname(__file__)


class WavegoRobot(LeggedRobot):
    """
    The robot env
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.sim_device = sim_device
        if self.cfg.customize.debugger_mode != 'none':
            self.debugger = TransitionDebugger(mode=self.cfg.customize.debugger_mode,
                                               sequence_len=self.cfg.customize.debugger_sequence_len,
                                               transition_path=os.path.join(current_folder, 'data'))
        self.num_envs = self.cfg.env.num_envs
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

    def update_and_get_state_mem(self, s_roll, c_roll, s_pitch, c_pitch):

        self.mem["s_roll"].append(s_roll)
        self.mem["c_roll"].append(c_roll)
        self.mem["s_pitch"].append(s_pitch)
        self.mem["c_pitch"].append(c_pitch)
        self.mem["base_ang_vel"].append(self.base_ang_vel * self.obs_scales.ang_vel)
        if self.cfg.noise.add_noise:
            self.mem["actions"].append(self.actions + 0.05*torch.rand(self.cfg.env.num_envs, 12).cuda())
        else:
            self.mem["actions"].append(self.actions)

        result = {}
        for key in self.mem.keys():
            result[key] = self.mem[key][-1]
            for i in range(self.obs_mem_len-1):
                result[key] = torch.concat((result[key], self.mem[key][-1 - self.obs_mem_skip*i - self.obs_mem_skip]), 1)
        return result

    def compute_observations(self):
        """
        Computes observations
        """
        # time.sleep(0.1)
        # my observations
        dof_vel_from_deviation = (self.dof_pos - self.last_dof_pos) / self.dt
        roll, pitch, _ = get_euler_xyz(self.base_quat)
        if self.cfg.noise.add_noise:
            # if add noise
            roll += 0.1 * torch.rand(self.cfg.env.num_envs, ).cuda()
            pitch += 0.1 * torch.rand(self.cfg.env.num_envs, ).cuda()
        s_roll, c_roll, s_pitch, c_pitch = torch.unsqueeze(torch.sin(roll), 1), torch.unsqueeze(torch.cos(roll), 1), \
                                           torch.unsqueeze(torch.sin(pitch), 1), torch.unsqueeze(torch.cos(pitch), 1)

        # use mem state
        if self.cfg.customize.use_state_mem:
            mem_obs = self.update_and_get_state_mem(s_roll, c_roll, s_pitch, c_pitch)

        # make obs_list
        obs_list = []
        for state in self.cfg.customize.observation_states:
            if state == 'angular_v':
                if self.cfg.customize.use_state_mem:
                    obs_list.append(mem_obs["base_ang_vel"])
                else:
                    obs_list.append(self.base_ang_vel * self.obs_scales.ang_vel)
            elif state == 'row_pitch':
                if self.cfg.customize.use_state_mem:
                    obs_list.append(mem_obs["s_roll"])
                    obs_list.append(mem_obs["c_roll"])
                    obs_list.append(mem_obs["s_pitch"])
                    obs_list.append(mem_obs["c_pitch"])
                else:
                    obs_list.append(s_roll)
                    obs_list.append(c_roll)
                    obs_list.append(s_pitch)
                    obs_list.append(c_pitch)
            elif state == 'top_commands':
                obs_list.append(self.commands[:, :3] * self.commands_scale)
            elif state == 'dof_pos':
                obs_list.append((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)
            elif state == 'dof_vel':
                obs_list.append(dof_vel_from_deviation * self.obs_scales.dof_vel)
            elif state == 'dof_action':
                if self.cfg.customize.use_state_mem:
                    obs_list.append(mem_obs["actions"])
                else:
                    obs_list.append(self.actions)
            elif state == 'targets':
                obs_list.append(self.targets)
            elif state == 'sequence_dof_pos':
                obs_list.extend(list(self.sequence_dof_pos))
            elif state == 'sequence_dof_action':
                obs_list.extend(list(self.sequence_dof_action))
            elif state == 'env_factor':
                for factor in self.cfg.customize.env_factors:
                    if factor == 'payload':
                        obs_list.append(torch.squeeze(self.payloads, -1))
                    elif factor == 'dof_stiffness':
                        obs_list.append(self.p_gains)
                    elif factor == 'dof_damping':
                        obs_list.append(self.d_gains)
                    elif factor == 'terrain_friction':
                        obs_list.append(torch.squeeze(self.friction_coeffs, -1))
                    elif factor == 'inclination':
                        obs_list.append(self.inclination)
                    elif factor == 'restitution':
                        obs_list.append(torch.squeeze(self.restitution, -1))

        self.obs_buf = torch.cat(obs_list, dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

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
        # refresh some buffers
        forward = quat_apply(self.base_quat, self.forward_vec)
        self.heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.inclination = torch.stack(
            [torch.cos(self.heading) * np.deg2rad(self.cfg.customize.plane_tilted_angle),
             -torch.sin(self.heading) * np.deg2rad(self.cfg.customize.plane_tilted_angle)],
            dim=1)

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

    def _reward_energy(self):
        return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)

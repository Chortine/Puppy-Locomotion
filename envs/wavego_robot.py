from legged_gym.envs import LeggedRobot
import torch
import numpy as np


class WavegoRobot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

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

    def compute_observations(self):
        """ Computes observations
        """
        # my observations
        self.dof_vel_from_deviation = (self.dof_pos - self.last_dof_pos) / self.dt
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.projected_gravity,  # 3
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel_from_deviation * self.obs_scales.dof_vel,
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

        self.last_dof_pos[:] = self.dof_pos[:]
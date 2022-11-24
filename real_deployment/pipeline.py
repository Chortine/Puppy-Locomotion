import numpy as np
from RPi.state import State
from real_deployment.wavego_flat_config import WavegoFlatCfg, WavegoFlatCfgPPO
import torch
from collections import deque
from torch_utils import get_euler_xyz

# some leg constants # in millimeter
Linkage_S = 12.2
Linkage_A = 40.0
Linkage_B = 40.0
Linkage_C = 39.8153
Linkage_D = 31.7750
Linkage_E = 30.8076


class RealPipeline:
    """
    Handling the data transformation between sim and real during the agent running pipeline
    """

    def __init__(self, env_cfg: WavegoFlatCfg, train_cfg: WavegoFlatCfgPPO):
        self.cfg = env_cfg
        self.train_cfg = train_cfg
        self.sim_joint_name_list = ['fl_j0', 'fl_j1', 'fl_j2',
                                    'fr_j0', 'fr_j1', 'fr_j2',
                                    'rl_j0', 'rl_j1', 'rl_j2',
                                    'rr_j0', 'rr_j1', 'rr_j2']
        self.default_joint_angles = self.cfg.init_state.default_joint_angles
        self.sim_joint_cmd_dict = {key: [self.default_joint_angles[key], self.default_joint_angles[key]] for key in
                                   self.sim_joint_name_list}
        self.dt = 0.05  # TODO: do we need this smaller dt to obtain finer observation?
        self.duration_rl = 0.1
        self.last_clipped_action = torch.zeros(1, len(self.sim_joint_name_list))
        # self.last_clipped_action = [0.0 for joint in self.sim_joint_name_list]

        # ====================== INIT MEMORY =========================
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
            self.mem["s_roll"].append(torch.zeros(1, 1))
            self.mem["c_roll"].append(torch.ones(1, 1))
            self.mem["s_pitch"].append(torch.zeros(1, 1))
            self.mem["c_pitch"].append(torch.ones(1, 1))
            self.mem["base_ang_vel"].append(torch.zeros(1, 3))
            self.mem["actions"].append(torch.zeros(1, 12))

        # ================ init rma_obs_memory ========================= #
        self.rma_obs_mem = deque(maxlen=self.cfg.customize.rma_obs_mem_len)
        if 'rma_obs_mem' in self.cfg.customize.obs_shape_dict.keys():
            for i in range(self.cfg.customize.rma_obs_mem_len):
                self.rma_obs_mem.append(torch.zeros(1, self.cfg.customize.obs_shape_dict['rma_obs_mem'][0]))

    # def obs_dict_to_tensor(self, obs_dict):
    #     """
    #
    #     :param obs_dict:
    #     :return:
    #     """
    #     # make torch tensor and add batch dimension
    #     base_ang_vel = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['base_ang_vel'])), dim=0)
    #     commands = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['commands'])), dim=0)
    #     dof_pos = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['dof_pos'])), dim=0)
    #     dof_vel = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['dof_vel'])), dim=0)
    #     actions = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['actions'])), dim=0)
    #
    #     obs_tensor = torch.cat((base_ang_vel,
    #                             commands,
    #                             dof_pos,
    #                             dof_vel,
    #                             actions),
    #                            dim=-1
    #                            )
    #     # check dimension
    #     assert obs_tensor.size()[-1] == self.cfg.env.num_observations, f'require input size ' \
    #                                                                    f'{self.cfg.env.num_observations}, ' \
    #                                                                    f'but receive {obs_tensor.size()[-1]}'
    #     return obs_tensor

    def update_and_get_state_mem(self,
                                 s_roll,
                                 c_roll,
                                 s_pitch,
                                 c_pitch,
                                 base_ang_vel,
                                 actions
                                 ):

        self.mem["s_roll"].append(s_roll)
        self.mem["c_roll"].append(c_roll)
        self.mem["s_pitch"].append(s_pitch)
        self.mem["c_pitch"].append(c_pitch)
        self.mem["base_ang_vel"].append(base_ang_vel * self.cfg.normalization.obs_scales.ang_vel)
        self.mem["actions"].append(actions)

        result = {}
        for key in self.mem.keys():
            result[key] = self.mem[key][-1]
            for i in range(self.obs_mem_len - 1):
                result[key] = torch.concat((result[key], self.mem[key][-1 - self.obs_mem_skip * i - self.obs_mem_skip]), 1)
        return result

    def state_to_obs(self, state: State):
        """
        Computes observations
        """
        # time.sleep(0.1)
        # my observations
        top_commands = torch.tensor([[0.2, 0, 0]], device='cpu')
        commands_scale = torch.tensor([self.cfg.normalization.obs_scales.lin_vel,
                                       self.cfg.normalization.obs_scales.lin_vel,
                                       self.cfg.normalization.obs_scales.ang_vel],
                                      device='cpu')

        # action should be the action that causes current observations
        actions = self.last_clipped_action
        base_ang_vel_real = np.deg2rad(state.angular_v)
        base_ang_vel_sim = torch.tensor([[-base_ang_vel_real[1], base_ang_vel_real[0], base_ang_vel_real[2]]])
        base_rp = np.deg2rad(state.rp)
        roll = torch.tensor([[base_rp[0]]])
        pitch = torch.tensor([[base_rp[1]]])

        # roll, pitch, _ = get_euler_xyz(self.base_quat)

        s_roll, c_roll, s_pitch, c_pitch = torch.sin(roll), torch.cos(roll), \
                                           torch.sin(pitch), torch.cos(pitch)

        # use mem state
        mem_obs = self.update_and_get_state_mem(s_roll,
                                                c_roll,
                                                s_pitch,
                                                c_pitch,
                                                base_ang_vel_sim,
                                                actions)

        # make obs_list
        obs_list = []
        obs_mem_list = []
        for state in self.cfg.customize.obs_shape_dict['obs_tensor']:
            if state == 'common_states':
                for common_state in self.cfg.customize.obs_shape_dict['obs_tensor']['common_states']:
                    if common_state == 'angular_v':
                        obs_list.append(mem_obs["base_ang_vel"])

                        obs_mem_list.append(base_ang_vel_sim * self.cfg.normalization.obs_scales.ang_vel)
                    elif common_state == 'row_pitch':
                        obs_list.append(mem_obs["s_roll"])
                        obs_list.append(mem_obs["c_roll"])
                        obs_list.append(mem_obs["s_pitch"])
                        obs_list.append(mem_obs["c_pitch"])

                        obs_mem_list.append(s_roll)
                        obs_mem_list.append(c_roll)
                        obs_mem_list.append(s_pitch)
                        obs_mem_list.append(c_pitch)
                    elif common_state == 'top_commands':
                        obs_list.append(top_commands[:, :3] * commands_scale)

                        obs_mem_list.append(top_commands[:, :3] * commands_scale)
                    elif common_state == 'dof_action':
                        obs_list.append(mem_obs["actions"])

                        obs_mem_list.append(actions)
            elif state == 'env_factor':
                fake_env_factors = torch.zeros(1, self.cfg.customize.obs_shape_dict['obs_tensor']['env_factor'])
                obs_list.append(fake_env_factors)

        obs_buf = torch.cat(obs_list, dim=-1)
        obs_mem_buf = torch.cat(obs_mem_list, dim=-1)
        if 'rma_obs_mem' in self.cfg.customize.obs_shape_dict.keys():
            self.rma_obs_mem.append(obs_mem_buf)

        obs_dict = {
            'obs_tensor': obs_buf.float()}
        if 'rma_obs_mem' in self.cfg.customize.obs_shape_dict.keys():
            obs_dict.update({'rma_obs_mem': torch.stack(list(self.rma_obs_mem), dim=-1).float()})

        return obs_dict

    # def state_to_obs(self, state: State):
    #     """
    #     transfer the real robot states to network input
    #     :param state:
    #     :return:
    #     """
    #     obs_dict = {}
    #     # TODO: to check the zero point of the cmd angle
    #     base_ang_vel_real = np.deg2rad(state.angular_v)
    #     base_ang_vel_sim = np.asarray([-base_ang_vel_real[1], base_ang_vel_real[0], base_ang_vel_real[2]])
    #     top_commands = np.asarray([0.4, 0, 0])
    #     dof_pos_from_cmd = np.asarray(
    #         [self.sim_joint_cmd_dict[joint][-1] - self.sim_joint_cmd_dict[joint][0] for joint in
    #          self.sim_joint_name_list])
    #     dof_vel_from_pos = np.asarray(
    #         [(self.sim_joint_cmd_dict[joint][-1] - self.sim_joint_cmd_dict[joint][-2]) / self.duration_rl
    #          for joint in self.sim_joint_name_list])
    #
    #     # scaling
    #     base_ang_vel_sim *= self.cfg.normalization.obs_scales.ang_vel
    #     top_commands *= [self.cfg.normalization.obs_scales.lin_vel,
    #                      self.cfg.normalization.obs_scales.lin_vel,
    #                      self.cfg.normalization.obs_scales.ang_vel]
    #     dof_pos_from_cmd *= self.cfg.normalization.obs_scales.dof_pos
    #     dof_vel_from_pos *= self.cfg.normalization.obs_scales.dof_vel
    #
    #     obs_dict['base_ang_vel'] = base_ang_vel_sim
    #     obs_dict['commands'] = top_commands
    #     obs_dict['dof_pos'] = dof_pos_from_cmd
    #     obs_dict['dof_vel'] = dof_vel_from_pos
    #     obs_dict['actions'] = self.last_clipped_action
    #     obs_tensor = self.obs_dict_to_tensor(obs_dict)
    #
    #     return obs_tensor

    def compute_forward_kinematics_one_leg(self, angle_j0, angle_j1, angle_j2):
        """
        :param angle_j0: in radians
        :param angle_j1:
        :param angle_j2:
        :return: wiggle_angle[degrees], l2_end_x_leg[mm], l2_end_y_leg[mm]
        """
        # output wiggle angle and xy on the plane (x y are expressed in the leg coordinate)
        # wiggle_angle: degree; xy: millimeter
        wiggle_angle = np.rad2deg(angle_j0)
        # compute xy on the plane
        # angle in leg coordinate
        drift_j1 = np.deg2rad(45 + 90)
        angle_j1_leg = angle_j1 + drift_j1
        l1_end_x = Linkage_A * np.cos(angle_j1_leg)
        l1_end_y = Linkage_A * np.sin(angle_j1_leg)

        # calf link
        angle_calf = np.arctan2(Linkage_E, (Linkage_C + Linkage_D))
        len_calf = np.sqrt(Linkage_E ** 2 + (Linkage_C + Linkage_D) ** 2)

        drift_j2 = np.deg2rad(75) + angle_calf
        angle_j2_leg = angle_j2 + drift_j2
        angle_j2_leg = angle_j2_leg - (np.pi - angle_j1_leg)
        l2_end_x_l1 = len_calf * np.cos(angle_j2_leg)
        l2_end_y_l1 = len_calf * np.sin(angle_j2_leg)

        l2_end_x_leg = l2_end_x_l1 + l1_end_x
        l2_end_y_leg = l2_end_y_l1 + l1_end_y

        l2_end_x_leg -= Linkage_S / 2

        return wiggle_angle, l2_end_x_leg, l2_end_y_leg

    def action_to_command(self, clipped_rl_action, fix_j0=False):
        """
        compute the target joint angle in way similar to the sim
        the rl_action should already be clipped and scaled
        :param scaled_rl_action: a numpy array
        ['fl_j0', 'fl_j1', 'fl_j2',
        'fr_j0', 'fr_j1', 'fr_j2',
        'rl_j0', 'rl_j1', 'rl_j2',
        'rr_j0', 'rr_j1', 'rr_j2']
        """
        # default_joint_angles = self.cfg.init_state.default_joint_angles
        # clipped_rl_action = np.squeeze(clipped_rl_action)
        # print(f'clipped_rl_action {clipped_rl_action}')
        self.last_clipped_action = clipped_rl_action
        target_joint_angles = {}
        clipped_rl_action = np.squeeze(clipped_rl_action)
        for i, joint in enumerate(self.sim_joint_name_list):
            target_joint_angles[joint] = clipped_rl_action[i] * self.cfg.control.action_scale + \
                                         self.default_joint_angles[joint]
        leg_command_dict = {}
        for leg in ['fl', 'fr', 'rl', 'rr']:
            a, x, y = self.compute_forward_kinematics_one_leg(target_joint_angles[f'{leg}_j0'],
                                                              target_joint_angles[f'{leg}_j1'],
                                                              target_joint_angles[f'{leg}_j2'])
            # add to cmd memory
            self.sim_joint_cmd_dict[f'{leg}_j0'].append(target_joint_angles[f'{leg}_j0'])
            self.sim_joint_cmd_dict[f'{leg}_j1'].append(target_joint_angles[f'{leg}_j1'])
            self.sim_joint_cmd_dict[f'{leg}_j2'].append(target_joint_angles[f'{leg}_j2'])
            if fix_j0:
                a = 4  # fix to 4 degree
            leg_command_dict[f'{leg}_leg'] = [a, x, y]
        # alias
        cmd = leg_command_dict
        commands = [cmd['rr_leg'],
                    cmd['fr_leg'],
                    cmd['fl_leg'],
                    cmd['rl_leg']]
        commands = np.concatenate(commands)
        return commands

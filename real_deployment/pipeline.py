import numpy as np
from RPi.state import State
from real_deployment.base_legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch

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

    def __init__(self, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.sim_joint_name_list = ['fl_j0', 'fl_j1', 'fl_j2',
                                    'fr_j0', 'fr_j1', 'fr_j2',
                                    'rl_j0', 'rl_j1', 'rl_j2',
                                    'rr_j0', 'rr_j1', 'rr_j2']
        self.default_joint_angles = self.env_cfg.init_state.default_joint_angles
        self.sim_joint_cmd_dict = {key: [self.default_joint_angles[key], self.default_joint_angles[key]] for key in
                                   self.sim_joint_name_list}
        self.dt = 0.05  # TODO: do we need this smaller dt to obtain finer observation?
        self.duration_rl = 0.1
        self.last_scaled_actions = [0.0 for joint in self.sim_joint_name_list]

    def obs_dict_to_tensor(self, obs_dict):
        """

        :param obs_dict:
        :return:
        """
        # make torch tensor and add batch dimension
        base_ang_vel = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['base_ang_vel'])), dim=0)
        commands = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['commands'])), dim=0)
        dof_pos = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['dof_pos'])), dim=0)
        dof_vel = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['dof_vel'])), dim=0)
        actions = torch.unsqueeze(torch.tensor(np.squeeze(obs_dict['actions'])), dim=0)

        obs_tensor = torch.cat((base_ang_vel,
                                commands,
                                dof_pos,
                                dof_vel,
                                actions),
                               dim=-1
                               )
        # check dimension
        assert obs_tensor.size()[-1] == self.env_cfg.env.num_observations, f'require input size ' \
                                                                           f'{self.env_cfg.env.num_observations}, ' \
                                                                           f'but receive {obs_tensor.size()[-1]}'
        return obs_tensor

    def state_to_obs(self, state: State):
        """
        transfer the real robot states to network input
        :param state:
        :return:
        """
        obs_dict = {}
        # TODO: to check the zero point of the cmd angle
        base_ang_vel_real = np.deg2rad(state.angular_v)
        base_ang_vel_sim = np.asarray([-base_ang_vel_real[1], base_ang_vel_real[0], base_ang_vel_real[2]])
        top_commands = np.asarray([0.4, 0, 0])
        dof_pos_from_cmd = np.asarray([self.sim_joint_cmd_dict[joint][-1] - self.sim_joint_cmd_dict[joint][0] for joint in
                            self.sim_joint_name_list])
        dof_vel_from_pos = np.asarray([(self.sim_joint_cmd_dict[joint][-1] - self.sim_joint_cmd_dict[joint][-2]) / self.duration_rl
                            for joint in self.sim_joint_name_list])

        # scaling
        base_ang_vel_sim *= self.env_cfg.normalization.obs_scales.ang_vel
        top_commands *= [self.env_cfg.normalization.obs_scales.lin_vel,
                         self.env_cfg.normalization.obs_scales.lin_vel,
                         self.env_cfg.normalization.obs_scales.ang_vel]
        dof_pos_from_cmd *= self.env_cfg.normalization.obs_scales.dof_pos
        dof_vel_from_pos *= self.env_cfg.normalization.obs_scales.dof_vel

        obs_dict['base_ang_vel'] = base_ang_vel_sim
        obs_dict['commands'] = top_commands
        obs_dict['dof_pos'] = dof_pos_from_cmd
        obs_dict['dof_vel'] = dof_vel_from_pos
        obs_dict['actions'] = self.last_scaled_actions
        obs_tensor = self.obs_dict_to_tensor(obs_dict)

        return obs_tensor

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

    def action_to_command(self, scaled_rl_action, fix_j0=False):
        """
        compute the target joint angle in way similar to the sim
        the rl_action should already be clipped and scaled
        :param scaled_rl_action: a numpy array
        ['fl_j0', 'fl_j1', 'fl_j2',
        'fr_j0', 'fr_j1', 'fr_j2',
        'rl_j0', 'rl_j1', 'rl_j2',
        'rr_j0', 'rr_j1', 'rr_j2']
        """
        # default_joint_angles = self.env_cfg.init_state.default_joint_angles
        scaled_rl_action = np.squeeze(scaled_rl_action)
        print(f'scaled_rl_action {scaled_rl_action}')
        self.last_scaled_actions = scaled_rl_action
        target_joint_angles = {}
        for i, joint in enumerate(self.sim_joint_name_list):
            target_joint_angles[joint] = scaled_rl_action[i] + self.default_joint_angles[joint]
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

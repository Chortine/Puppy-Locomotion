import time
import os
from collections import deque
import statistics

import torch

# from rsl_rl.algorithms import PPO
# from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

from rl_utils.algorithms import PPO
from rl_utils.modules import ActorCritic

import os
import copy
import torch
import numpy as np
import random
from typing import Tuple
from real_deployment.wavego_flat_config import WavegoFlatCfg, WavegoFlatCfgPPO
from real_deployment.helpers import class_to_dict


def get_obs_len(obs_shape_dict: dict):
    value_sum = 0
    if not isinstance(obs_shape_dict, dict):
        return obs_shape_dict
    for key, value in obs_shape_dict.items():
        if isinstance(value, dict):
            value_sum += get_obs_len(value)
        else:
            if not isinstance(value, int):
                return value
            value_sum += value
    return value_sum


class RealRunner:
    """
    to write
    """

    def __init__(self,
                 env_cfg: WavegoFlatCfg,
                 train_cfg: dict,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device

        if env_cfg.env.num_privileged_obs is not None:
            num_critic_obs = env_cfg.env.num_privileged_obs
        else:
            num_critic_obs = env_cfg.env.num_observations

        actor_critic: ActorCritic = ActorCritic(env_cfg.customize.obs_shape_dict,
                                                num_critic_obs,
                                                env_cfg.env.num_actions,
                                                **self.policy_cfg).to(self.device)
        self.alg: PPO = PPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        obs_group_shape = {'obs_tensor': (get_obs_len(env_cfg.customize.obs_shape_dict['obs_tensor']),)}
        if 'rma_obs_mem' in env_cfg.customize.obs_shape_dict.keys():
            obs_group_shape.update({'rma_obs_mem': env_cfg.customize.obs_shape_dict['rma_obs_mem']})
        self.alg.init_storage(env_cfg.env.num_envs, self.num_steps_per_env, obs_group_shape,
                              [env_cfg.env.num_privileged_obs], [env_cfg.env.num_actions])

        # Log
        self.env_cfg = env_cfg

    def load(self, path, load_optimizer=True):
        """

        :param path:
        :param load_optimizer:
        :return:
        """
        loaded_dict = torch.load(path, map_location=torch.device('cpu'))
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """

        :param device:
        :return:
        """
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference


class RealAgent:
    """
    to write
    """

    def __init__(self, default_env_cfg: WavegoFlatCfg, default_train_cfg: WavegoFlatCfgPPO):
        self.default_env_cfg = default_env_cfg
        self.default_train_cfg = default_train_cfg
        # copy seed
        self.default_env_cfg.seed = self.default_train_cfg.seed
        self.policy = None

    def get_default_cfgs(self) -> Tuple[WavegoFlatCfg, WavegoFlatCfgPPO]:
        """

        :return:
        """
        train_cfg = self.default_train_cfg
        env_cfg = self.default_env_cfg
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def get_alg_runner(self, env_cfg=None, train_cfg=None) -> RealRunner:
        """

        :param env_cfg:
        :param train_cfg:
        :param log_root:
        :return:
        """
        # get env cfg
        if env_cfg is None:
            env_cfg = self.default_env_cfg

        # get train cfg
        if train_cfg is None:
            train_cfg = self.default_train_cfg

        train_cfg_dict = class_to_dict(train_cfg)
        runner = RealRunner(env_cfg, train_cfg_dict, device='cpu')
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        return runner

    def build_policy(self, ckpt_path, env_cfg=None, train_cfg=None):
        """

        :param ckpt_path:
        :param env_cfg:
        :param train_cfg:
        """
        ppo_runner = self.get_alg_runner(env_cfg=env_cfg, train_cfg=train_cfg)
        ppo_runner.load(ckpt_path)

        policy = ppo_runner.get_inference_policy(device='cpu')
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.policy = policy

    def inference_one_step(self, obs):
        """
        :param obs:
        :return:
        """
        actions: torch.Tensor
        # obs = obs.float()
        actions = self.policy(obs)
        actions = actions.detach().cpu()
        # clip and scale the action
        clip_actions = self.env_cfg.normalization.clip_actions
        clipped_actions = np.clip(actions, -clip_actions, clip_actions)
        # scaled_actions = clipped_actions * self.env_cfg.control.action_scale
        # scaled_actions = clipped_actions * 0.25
        # print(f'action_scale {self.env_cfg.control.action_scale}')
        return clipped_actions


if __name__ == '__main__':
    from real_deployment.wavego_flat_config import WavegoFlatCfg, WavegoFlatCfgPPO

    agent = RealAgent(default_env_cfg=WavegoFlatCfg(), default_train_cfg=WavegoFlatCfgPPO())
    # modify some configs here
    env_cfg, train_cfg = agent.get_default_cfgs()
    train_cfg.runner.resume = False
    ckpt_path = '/home/tianchu/Documents/code_qy/puppy-gym/logs/Oct12_17-15-27_run1/model_2000.pt'
    # load policy
    agent.build_policy(ckpt_path, env_cfg, train_cfg)

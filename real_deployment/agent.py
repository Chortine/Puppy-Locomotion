import time
import os
from collections import deque
import statistics

import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

import os
import copy
import torch
import numpy as np
import random
from typing import Tuple
from real_deployment.base_legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from real_deployment.helpers import class_to_dict


class RealRunner:
    """
    to write
    """

    def __init__(self,
                 env_cfg: LeggedRobotCfg,
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

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(env_cfg.env.num_observations,
                                                       num_critic_obs,
                                                       env_cfg.env.num_actions,
                                                       **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(env_cfg.env.num_envs, self.num_steps_per_env, [env_cfg.env.num_observations],
                              [env_cfg.env.num_privileged_obs], [env_cfg.env.num_actions])

        # Log
        self.env_cfg = env_cfg
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def log(self, locs, width=80, pad=35):
        """

        :param locs:
        :param width:
        :param pad:
        """
        self.tot_timesteps += self.num_steps_per_env * self.env_cfg.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env_cfg.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

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
        self.current_learning_iteration = loaded_dict['iter']
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

    def __init__(self, default_env_cfg: LeggedRobotCfg, default_train_cfg: LeggedRobotCfgPPO):
        self.default_env_cfg = default_env_cfg
        self.default_train_cfg = default_train_cfg
        # copy seed
        self.default_env_cfg.seed = self.default_train_cfg.seed
        self.policy = None

    def get_default_cfgs(self) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
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
        obs = obs.float()
        actions = self.policy(obs)
        actions = actions.detach().cpu()
        # clip and scale the action
        clip_actions = self.env_cfg.normalization.clip_actions
        clipped_actions = np.clip(actions, -clip_actions, clip_actions)
        # scaled_actions = clipped_actions * self.env_cfg.control.action_scale
        scaled_actions = clipped_actions * 0.15
        print(f'action_scale {self.env_cfg.control.action_scale}')
        return scaled_actions


if __name__ == '__main__':
    from real_deployment.wavego_flat_config import WavegoFlatCfg, WavegoFlatCfgPPO

    agent = RealAgent(default_env_cfg=WavegoFlatCfg(), default_train_cfg=WavegoFlatCfgPPO())
    # modify some configs here
    env_cfg, train_cfg = agent.get_default_cfgs()
    train_cfg.runner.resume = False
    ckpt_path = '/home/tianchu/Documents/code_qy/puppy-gym/logs/Oct12_17-15-27_run1/model_2000.pt'
    # load policy
    agent.build_policy(ckpt_path, env_cfg, train_cfg)


from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.register import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch


# # ===== stiffness ====== #
# # fore legs
# fore_j0_p = 2.0
# fore_j1_p = 0.4
# fore_j2_p = 0.5
#
# # hind legs
# hind_j0_p = 2.0
# hind_j1_p = 0.5
# hind_j2_p = 0.8
#
# # ===== damping ======= #
# # fore legs
# fore_j0_d = 0.01
# fore_j1_d = 0.01
# fore_j2_d = 0.01
#
# # hind legs
# hind_j0_d = 0.01
# hind_j1_d = 0.01
# hind_j2_d = 0.01

def choose_one_set_of_pds(log=True):
    # now we use randomly sampled strategy
    # pd order
    # 'j0_p',
    # 'j1_p',
    # 'j2_p',
    # 'j0_d',
    # 'j1_d',
    # 'j2_d'
    # pd_limits = [
    #     [0.01, 200],
    #     [0.01, 200],
    #     [0.01, 200],
    #     [0.001, 20],
    #     [0.001, 20],
    #     [0.001, 20],
    # ]
    pd_limits = [
        [-200, 200],
        [-200, 200],
        [-200, 200],
        [-20, 20],
        [-20, 20],
        [-20, 20],
    ]
    if log:
        log_pd_limits = np.log(pd_limits)
    else:
        log_pd_limits = np.asarray(pd_limits)
    # random sample in log space
    # log_pds = np.random.uniform(low=log_pd_limits[:, 0], high=log_pd_limits[:, 1])

    random_p = np.random.uniform(low=log_pd_limits[0, 0], high=log_pd_limits[0, 1])
    random_d = np.random.uniform(low=log_pd_limits[3, 0], high=log_pd_limits[3, 1])
    log_pds = np.asarray([random_p, random_p, random_p, random_d, random_d, random_d])

    if log:
        pds = np.exp(log_pds)
    else:
        pds = log_pds
    return pds


def search_pd(args):
    # args.task = "a1_flat"
    args.task = "wavego_flat"
    args.num_envs = 1000
    # pd_dict = {
    #     'j0_p',
    #     'j1_p',
    #     'j2_p',
    #     'j0_d',
    #     'j1_d',
    #     'j2_d'
    # }
    pd_all_envs = []
    for i in range(args.num_envs):
        pd_all_envs.append(choose_one_set_of_pds(log=False))
    pd_all_envs = np.asarray(pd_all_envs)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.customize.pd_all_envs = pd_all_envs
    error_all_envs = play(args, env_cfg, train_cfg)
    print(error_all_envs)
    p_for_envs = pd_all_envs[:, 0]
    d_for_envs = pd_all_envs[:, 3]
    plot_error_scatter(p_for_envs, d_for_envs, error_all_envs, log=False)
    amin = np.argmin(error_all_envs)
    print(f'best p is: {p_for_envs[amin]}, best d is: {d_for_envs[amin]}, error is: {error_all_envs[amin]}')

def play(args, env_cfg, train_cfg):
    ckpt_path = '/home/tianchu/Documents/code_qy/puppy-gym/logs/model_2000.pt'
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner.load(ckpt_path)

    policy = ppo_runner.get_inference_policy(device=env.device)

    while not env.common_step_counter == 200:
    # while not env.debugger.replay_done:
    # for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

    errors = env.mean_dof_errors.detach().cpu().numpy()
    # responses = env.responses

    return errors

def plot_error_scatter(p_for_envs, d_for_envs, errors, log=True):
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    if log:
        ax.scatter3D(np.log(p_for_envs), np.log(d_for_envs), errors)
    else:
        ax.scatter3D(p_for_envs, d_for_envs, errors)
    # Show plot
    ax.set_xlabel('p_for_envs')
    ax.set_ylabel('d_for_envs')
    ax.set_zlabel('errors')
    plt.show()

if __name__ == '__main__':
    args = get_args()
    search_pd(args)

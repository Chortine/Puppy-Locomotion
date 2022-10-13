from locale import currency
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from copy import copy

def meta_tune(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # init parameter theta
    theta_dim = 6  
    iterations = 50
    num_paths = 50
    elite_ratio = 0.2
    elite_num = int(num_paths * elite_ratio)
    initial_mean = np.concatenate((np.random.uniform(0, 1000, (num_paths, 3)), np.random.uniform(0, 200, (num_paths, 3))),axis=1)
    initial_std = np.concatenate((np.random.uniform(0, 50, (num_paths, 3)), np.random.uniform(0, 20, (num_paths, 3))),axis=1)

    theta = np.random.normal(initial_mean, initial_std)
    theta = np.clip(theta, a_min=0, a_max=5000)


    #init data buffer for update 
    elite_theta = np.zeros((elite_num, theta_dim))
    mean_info = np.zeros((iterations, theta_dim))
    std_info = np.zeros((iterations, theta_dim))
    path_reward_info = np.zeros(num_paths)
    max_reward_info = np.zeros(iterations)
    min_reward_info = np.zeros(iterations)
    mean_reward_info = np.zeros(iterations)
    
    mean_info[0,:] = np.mean(theta, axis=0)
    std_info[0,:] = np.std(theta, ddof=1 ,axis=0) 

    # override some cfg parameters
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = '/home/jackie/code/legged_gym/logs/test/qa'
    train_cfg.runner.checkpoint = 14000

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner2(env=env, name=args.task, args=args, train_cfg=train_cfg, env_cfg=env_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    logger = Logger(env.dt)
    stop_rew_log = env.max_episode_length + 1

    dof_props = env.gym.get_asset_dof_properties(env.robot_asset).copy()
    for iteration in tqdm(range(iterations)):
        for path in range(num_paths):
            # current_pd_parameter = theta[path,:]
            current_pd_parameter = [202.06  ,6.49, 13.52 ,196.52, 100.89, 113.04]
            env_cfg.control.stiffness = {
                "hips_leg_hip1_jnt1_joint": current_pd_parameter[0],
                "hip1_jnt1_leg1_jnt1_joint": current_pd_parameter[1],
                "leg1_jnt1_knee1_jnt1_joint": current_pd_parameter[2],
                "hips_leg_hip1_jnt2_joint": current_pd_parameter[0],
                "hip1_jnt2_leg1_jnt2_joint": current_pd_parameter[1],
                "leg1_jnt2_knee1_jnt2_joint": current_pd_parameter[2],
                "hips_leg_hip1_jnt3_joint": current_pd_parameter[0],
                "hip1_jnt3_leg1_jnt3_joint": current_pd_parameter[1],
                "leg1_jnt3_knee1_jnt3_joint": current_pd_parameter[2],
                "hips_leg_hip1_jnt4_joint": current_pd_parameter[0],
                "hip1_jnt4_leg1_jnt4_joint": current_pd_parameter[1],
                "leg1_jnt4_knee1_jnt4_joint": current_pd_parameter[2]
            }
            env_cfg.control.damping = {
                    "hips_leg_hip1_jnt1_joint": current_pd_parameter[3],
                    "hip1_jnt1_leg1_jnt1_joint": current_pd_parameter[4],
                    "leg1_jnt1_knee1_jnt1_joint": current_pd_parameter[5],
                    "hips_leg_hip1_jnt2_joint": current_pd_parameter[3],
                    "hip1_jnt2_leg1_jnt2_joint": current_pd_parameter[4],
                    "leg1_jnt2_knee1_jnt2_joint": current_pd_parameter[5],
                    "hips_leg_hip1_jnt3_joint": current_pd_parameter[3],
                    "hip1_jnt3_leg1_jnt3_joint": current_pd_parameter[4],
                    "leg1_jnt3_knee1_jnt3_joint": current_pd_parameter[5],
                    "hips_leg_hip1_jnt4_joint": current_pd_parameter[3],
                    "hip1_jnt4_leg1_jnt4_joint": current_pd_parameter[4],
                    "leg1_jnt4_knee1_jnt4_joint": current_pd_parameter[5]
                } 
            env.modify_all_dof_props(props=dof_props)

            logger.reset()
            logger.num_episodes = 0

            for i in range(int(env.max_episode_length + 2)):
                actions = policy(obs.detach())
                obs, _, rews, dones, infos = env.step(actions.detach())
                if  0 < i < stop_rew_log:
                    if infos["episode"]:
                        num_episodes = torch.sum(env.reset_buf).item()
                        if num_episodes>0:
                            logger.log_rewards(infos["episode"], num_episodes)
                elif i==stop_rew_log:
                    path_reward_info[path] = logger.get_rewards()


        #store info
        max_reward_info[iteration] = np.max(path_reward_info) 
        min_reward_info[iteration] = np.min(path_reward_info) 
        mean_reward_info[iteration] = np.mean(path_reward_info)
        #select elite
        theta_and_reward = list(zip(np.round(theta, 2), np.round(path_reward_info, 2)))
        theta_and_reward_sorted = sorted(theta_and_reward, key=lambda x:x[1], reverse=True)
        for index, value in enumerate(theta_and_reward_sorted):
            if index == elite_num:
                break
            elite_theta[index,:] = value[0]
        # policy update using CEM
        new_mean = np.mean(elite_theta, axis=0)
        mean_info[iteration,:] = new_mean
        new_mean = np.tile(new_mean,(num_paths,1))
        new_std = np.std(elite_theta, ddof=1 ,axis=0) 
        std_info[iteration,:] = new_std
        new_std = np.tile(new_std,(num_paths,1))
        theta = np.random.normal(new_mean, new_std)
        theta = np.clip(theta, a_min=0, a_max=5000)
        print('Iteration:%d'%(iteration + 1) + ' '+'AverageReward:%.2f'%(mean_reward_info[iteration]))
        print('-mean:', np.round(mean_info[iteration], 2))
        print('-std:', np.round(std_info[iteration], 2))
    np.save(os.path.join(log_dir, 'AverageReward'), mean_reward_info)
    np.save(os.path.join(log_dir, 'PD_parameter'), mean_info)

    plt.figure(1,figsize=(12.8,8))
    plt.plot(np.arange(1, iterations + 1), mean_reward_info[:], marker='v', color='red', label='Average')
    plt.plot(np.arange(1, iterations + 1), max_reward_info[:], marker='x', color='fuchsia', label='Max')
    plt.plot(np.arange(1, iterations + 1), min_reward_info[:], marker='o', color='peru', label='Min')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'Reward'))
    plt.close()

    plt.figure(2,figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(np.arange(1, iterations + 1), mean_info[:, 0], marker='v', color='darkviolet', label='hip')
    plt.plot(np.arange(1, iterations + 1), mean_info[:, 1], marker='x', color='blue', label='leg')
    plt.plot(np.arange(1, iterations + 1), mean_info[:, 2], marker='o', color='lightseagreen', label='knee')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Stiffness', fontsize=16)
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(np.arange(1, iterations + 1), mean_info[:, 3], marker='v', color='red', label='hip')
    plt.plot(np.arange(1, iterations + 1), mean_info[:, 4], marker='x', color='tomato', label='leg')
    plt.plot(np.arange(1, iterations + 1), mean_info[:, 5], marker='o', color='gold', label='knee')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Damping', fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'pd_parameter'))
    plt.close()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    args = get_args()
    meta_tune(args)

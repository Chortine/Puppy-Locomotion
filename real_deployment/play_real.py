import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from real_deployment.agent import RealAgent
from real_deployment.pipeline import RealPipeline
from real_deployment.wavego_flat_config import WavegoFlatCfg, WavegoFlatCfgPPO
from RPi.run import RPInterface
from collections import deque
import time 
import numpy as np

if __name__ == '__main__':
    agent = RealAgent(default_env_cfg=WavegoFlatCfg(), default_train_cfg=WavegoFlatCfgPPO())
    robot = RPInterface(config={"log": True})
    robot.light_ctrl("magenta")
    # modify some configs here
    env_cfg, train_cfg = agent.get_default_cfgs()
    train_cfg.runner.resume = False
    ckpt_path = '/home/tianchu/Desktop/logs/model_2000.pt'
    # load policy
    agent.build_policy(ckpt_path, env_cfg, train_cfg)
    pipline = RealPipeline(env_cfg, train_cfg)
    robot_state = robot.get_state()
    agent_obs = pipline.state_to_obs(robot_state)
    step_count = 0
    time_deque = deque(maxlen=50)
    sim_dt = env_cfg.sim.dt
    decimation = env_cfg.control.decimation
    # rl_step_interval = sim_dt * decimation
    rl_step_interval = 1 / 10
    # TODO: Frequency control
    while True:
        # i = 10
        # a = int(i)
        # RL_action = [a, a, a, a, a, a, a, a, a, a, a, a]
        # robot.step(RL_action, 'servo')
    
        start_time = time.time()
        step_count += 1
        print(f'@@@@@ step_count {step_count}')
        print(f'~~~~~~~~~~~~~~agent_obs: {agent_obs}~~~~~~~~~~~~~')
        scaled_rl_action = agent.inference_one_step(agent_obs)
        command = pipline.action_to_command(scaled_rl_action)
        print(f'===================== command: {command} =======================')
        one_leg_command = [5.0, 20, 100]
        # command = one_leg_command + one_leg_command + one_leg_command + one_leg_command
        command = list(np.asarray(command, dtype=float))
        robot.step(command, mode='toe_pose')  # first time don't do it
        
        robot_state = robot.get_state()
        agent_obs = pipline.state_to_obs(robot_state)
        time_to_sleep = rl_step_interval - (time.time()-start_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        time_deque.append(time.time()-start_time)
        print(f'Inference Frequency is {len(time_deque)/sum(time_deque)} // target: {1/rl_step_interval}')

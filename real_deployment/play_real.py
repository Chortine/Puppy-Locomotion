import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from real_deployment.agent import RealAgent
from real_deployment.pipeline import RealPipeline
from real_deployment.wavego_flat_config import WavegoFlatCfg, WavegoFlatCfgPPO
from collections import deque
import time
import numpy as np
from transition_debugger import TransitionDebugger
import torch

deployment_folder = os.path.dirname(__file__)
from multiprocessing import Process, Queue
from pynput import keyboard
# import keyboard

def get_keyboard_input(command_deque):
    print('Press a or d to continue:')
    print('loop in get_keyboard_input')
    with keyboard.Events() as events:
        while True:
            # Block for as much as possible
            event = events.get(1e6)
            if event.key == keyboard.KeyCode.from_char('a'):
                print("get_command_a")
                command_deque.put('a', block=False)
            elif event.key == keyboard.KeyCode.from_char('d'):
                print("get_command_d")
                command_deque.put('d', block=False)
            elif event.key == keyboard.KeyCode.from_char('w'):
                print("get_command_w")
                command_deque.put('w', block=False)
            elif event.key == keyboard.KeyCode.from_char('s'):
                print("get_command_s")
                command_deque.put('s', block=False)

def tune_top_command(top_commands, keyboard_deque):
    if keyboard_deque.empty():
        return top_commands
    else:
        keyboard_input = keyboard_deque.get(block=False)
        if keyboard_input == 'w':
            # top_commands=[0.2, 0, 0.05]
            top_commands[0] += 0.01
        elif keyboard_input == 's':
            top_commands[0] -= 0.01
        elif keyboard_input == 'a':
            top_commands[2] += 0.05
        elif keyboard_input == 'd':
            top_commands[2] -= 0.05
    # clip the commands
        top_commands = np.clip(top_commands, a_min=[0.0, 0, -0.2], a_max=[0.3, 0, 0.2])
        print(f'top_commands {top_commands}')
        return top_commands
    
def play_real():
    turn_command_deque = Queue(maxsize=10)
    # turn_command_deque.append(0)
    from RPi.run import RPInterface
    top_commands=[0.08, 0, 0.05]
    my_process = Process(target=get_keyboard_input, args=(turn_command_deque,))
    my_process.start()
    # my_process.join()
    debugger_mode = 'none'
    if debugger_mode != 'none':
        debugger = TransitionDebugger(mode=debugger_mode, sequence_len=100,
                                      transition_path=os.path.join(deployment_folder, 'data'))
    agent = RealAgent(default_env_cfg=WavegoFlatCfg(), default_train_cfg=WavegoFlatCfgPPO())
    robot = RPInterface(config={"log": True})
    # robot.light_ctrl("magenta")
    # modify some configs here
    env_cfg, train_cfg = agent.get_default_cfgs()
    train_cfg.runner.resume = False
    ckpt_path = os.path.join(deployment_folder, 'model/model_4000.pt')
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
    rl_step_interval = 1 / 30
    scaled_rl_action = torch.Tensor(np.zeros(12))
    command = [0] * 12
    # TODO: Frequency control
    while True:
        # i = 10
        # a = int(i)
        # RL_action = [a, a, a, a, a, a, a, a, a, a, a, a]
        # robot.step(RL_action, 'servo')

        start_time = time.time()
        step_count += 1
        # print(f'@@@@@ step_count {step_count}')
        # top_commands=[0.2, 0, 0.05]
        top_commands = tune_top_command(top_commands, turn_command_deque)
        # print(f'top_commands {top_commands}')
        # my_process.join()
        # print(f'~~~~~~~~~~~~~~agent_obs: {agent_obs}~~~~~~~~~~~~~')
        if scaled_rl_action is not None and debugger_mode != 'none':
            fake_obs = debugger.step(obs=agent_obs.detach().cpu(),
                                     action=np.squeeze(scaled_rl_action.detach().cpu().numpy()),
                                     toe_pose=command)
            agent_obs = torch.tensor(fake_obs, device='cpu')

        clipped_rl_action = agent.inference_one_step(agent_obs)
        command = pipline.action_to_command(clipped_rl_action, fix_j0=True)
        # print(f'===================== command: {command} =======================')
        # one_leg_command = [5.0, 20, 100]
        # command = one_leg_command + one_leg_command + one_leg_command + one_leg_command
        command = list(np.asarray(command, dtype=float))
        robot.step(command, mode='toe_pose')  # first time don't do it
        robot_state = robot.get_state()
        agent_obs = pipline.state_to_obs(robot_state, top_commands=top_commands)
        time_to_sleep = rl_step_interval - (time.time() - start_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        time_deque.append(time.time() - start_time)
        # print(f'Inference Frequency is {len(time_deque) / sum(time_deque)} // target: {1 / rl_step_interval}')

class FakeState:
    rp = [0., 0.]  # rad
    angular_v = [0.0, 0.0, 0.0]  # deg/second

def play_fake():
    debugger_mode = 'replay'
    if debugger_mode != 'none':
        debugger = TransitionDebugger(mode=debugger_mode, sequence_len=100,
                                      transition_path=os.path.join(deployment_folder, 'data'))
    agent = RealAgent(default_env_cfg=WavegoFlatCfg(), default_train_cfg=WavegoFlatCfgPPO())
    # robot.light_ctrl("magenta")
    # modify some configs here
    env_cfg, train_cfg = agent.get_default_cfgs()
    train_cfg.runner.resume = False
    ckpt_path = os.path.join(deployment_folder, 'model/model_4000.pt')
    # load policy
    agent.build_policy(ckpt_path, env_cfg, train_cfg)
    pipline = RealPipeline(env_cfg, train_cfg)
    robot_state = FakeState()
    agent_obs = pipline.state_to_obs(robot_state)
    step_count = 0
    time_deque = deque(maxlen=50)
    sim_dt = env_cfg.sim.dt
    decimation = env_cfg.control.decimation
    # rl_step_interval = sim_dt * decimation
    rl_step_interval = 1 / 10
    scaled_rl_action = torch.Tensor(np.zeros(12))
    command = [0] * 12
    # TODO: Frequency control
    while True:
        start_time = time.time()
        step_count += 1
        print(f'@@@@@ step_count {step_count}')
        # print(f'~~~~~~~~~~~~~~agent_obs: {agent_obs}~~~~~~~~~~~~~')
        clipped_rl_action = agent.inference_one_step(agent_obs)
        command = pipline.action_to_command(clipped_rl_action, fix_j0=True)
        # print(f'===================== command: {command} =======================')
        # one_leg_command = [5.0, 20, 100]
        # command = one_leg_command + one_leg_command + one_leg_command + one_leg_command
        command = list(np.asarray(command, dtype=float))
        # robot_state = FakeState()
        agent_obs = pipline.state_to_obs(robot_state)
        time_to_sleep = rl_step_interval - (time.time() - start_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        time_deque.append(time.time() - start_time)
        print(f'Inference Frequency is {len(time_deque) / sum(time_deque)} // target: {1 / rl_step_interval}')


if __name__ == '__main__':
    play_real()

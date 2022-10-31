import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
from real_deployment.pipeline import RealPipeline
from real_deployment.wavego_flat_config import WavegoFlatCfg, WavegoFlatCfgPPO


def main():
    with open('/home/tianchu/Documents/code_qy/puppy-gym/real_deployment/data/replay_transitions_1027_1.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    replay_obs_deque = data_dict['obs_deque']
    replay_action_deque = data_dict['action_deque']
    replay_toe_pose_deque = data_dict['toe_pose_deque']

    with open('/home/tianchu/Documents/code_qy/puppy-gym/real_deployment/data/replay_transitions.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    obs_deque = data_dict['obs_deque']
    action_deque = data_dict['action_deque']
    toe_pose_deque = data_dict['toe_pose_deque']

    pipeline = RealPipeline(WavegoFlatCfg, WavegoFlatCfgPPO)


    # plt.plot(np.asarray(action_deque)[:, 4], marker='.')
    # plt.plot(np.asarray(replay_action_deque)[:, 4])
    # plt.show()
    #
    # plt.plot(np.asarray(toe_pose_deque)[:, 4], marker='.')
    # plt.plot(np.asarray(action_deque)[:, 5])
    # plt.plot(np.asarray(replay_toe_pose_deque)[:, 4])
    # plt.show()

    # plt.plot(np.asarray(toe_pose_deque)[:, 7], marker='.')
    # plt.plot(np.asarray(action_deque)[:, 8])
    # plt.plot(np.asarray(replay_toe_pose_deque)[:, 7])
    # plt.show()

    # plt.subplot(311)
    # plt.plot(np.asarray(replay_action_deque)[:, 3], marker='.')
    # plt.plot(np.asarray(replay_action_deque)[:, 4])
    # plt.plot(np.asarray(replay_action_deque)[:, 5])
    # plt.subplot(312)
    # plt.plot(np.asarray(action_deque)[:100, 3], marker='.')
    # plt.plot(np.asarray(action_deque)[:100, 4])
    # # plt.plot(np.asarray(action_deque)[:100, 5])
    # plt.subplot(313)
    # plt.plot(np.asarray(toe_pose_deque)[:100, 4], marker='.')
    # plt.plot(np.asarray(replay_toe_pose_deque)[:100, 4])
    # plt.show()

    # plt.subplot(321)
    # plt.plot(np.asarray(replay_action_deque)[:, 3], marker='.')
    # plt.plot(np.asarray(replay_action_deque)[:, 4])
    # plt.plot(np.asarray(replay_action_deque)[:, 5])
    # plt.subplot(323)
    # plt.plot(np.asarray(action_deque)[:100, 3], marker='.')
    # plt.plot(np.asarray(action_deque)[:100, 4])
    # # plt.plot(np.asarray(action_deque)[:100, 5])
    # plt.subplot(325)
    # plt.plot(np.asarray(toe_pose_deque)[:100, 4], marker='.')
    # plt.plot(np.asarray(replay_toe_pose_deque)[:100, 4])
    # # plt.show()
    #
    # plt.subplot(322)
    # plt.plot(np.asarray(replay_action_deque)[:, 6], marker='.')
    # plt.plot(np.asarray(replay_action_deque)[:, 7])
    # plt.plot(np.asarray(replay_action_deque)[:, 8])
    # plt.subplot(324)
    # plt.plot(np.asarray(action_deque)[:100, 7], marker='.')
    # plt.plot(np.asarray(action_deque)[:100, 8])
    # # plt.plot(np.asarray(action_deque)[:100, 5])
    # plt.subplot(326)
    # plt.plot(np.asarray(toe_pose_deque)[:100, 7], marker='.')
    # plt.plot(np.asarray(replay_toe_pose_deque)[:100, 7])
    # plt.show()

    # """
    # ============= Back Legs
    # """
    # # rr leg
    # plt.subplot(221)
    # plt.plot(np.asarray(toe_pose_deque)[:100, 1], marker='.')
    # plt.plot(np.asarray(replay_toe_pose_deque)[:100, 1])
    # plt.subplot(223)
    # plt.plot(np.asarray(action_deque)[:100, 10], marker='.')
    # plt.plot(np.asarray(action_deque)[:100, 11])
    #
    # # rl leg
    # plt.subplot(222)
    # plt.plot(np.asarray(toe_pose_deque)[:100, 10], marker='.')
    # plt.plot(np.asarray(replay_toe_pose_deque)[:100, 10])
    # plt.subplot(224)
    # plt.plot(np.asarray(action_deque)[:100, 7], marker='.')
    # plt.plot(np.asarray(action_deque)[:100, 8])
    # plt.show()
    #
    # # action order:
    # # ['fl_j0', 'fl_j1', 'fl_j2',
    # #  'fr_j0', 'fr_j1', 'fr_j2',
    # #  'rl_j0', 'rl_j1', 'rl_j2',
    # #  'rr_j0', 'rr_j1', 'rr_j2']
    #
    """
    ============= Fore Legs
    """

    # fr leg
    plt.subplot(221)
    plt.plot(np.asarray(toe_pose_deque)[:100, 4], marker='.')
    plt.plot(np.asarray(replay_toe_pose_deque)[:100, 4])
    plt.subplot(223)
    plt.plot(np.asarray(action_deque)[:100, 4], marker='.')
    plt.plot(np.asarray(action_deque)[:100, 5])

    # fl leg
    plt.subplot(222)
    plt.plot(np.asarray(toe_pose_deque)[:100, 7], marker='.')
    plt.plot(np.asarray(replay_toe_pose_deque)[:100, 7])
    plt.subplot(224)
    plt.plot(np.asarray(action_deque)[:100, 1], marker='.')
    plt.plot(np.asarray(action_deque)[:100, 2])
    plt.show()

    commands = []
    for scaled_rl_action in action_deque:
        command = pipeline.action_to_command(scaled_rl_action, fix_j0=True)
        commands.append(command)

    plt.subplot(221)
    plt.plot(np.asarray(commands)[:100, 4], marker='.')
    plt.subplot(223)
    plt.plot(np.asarray(action_deque)[:100, 4])
    plt.plot(np.asarray(action_deque)[:100, 5])
    # fl leg
    plt.subplot(222)
    plt.plot(np.asarray(commands)[:100, 7], marker='.')
    plt.subplot(224)
    plt.plot(np.asarray(action_deque)[:100, 1], marker='.')
    plt.plot(np.asarray(action_deque)[:100, 2])
    plt.show()
    print('end')

def try_plot():
    # Isaac
    # gym: 1.3e-3 （都开一个环境）
    # Isaac
    # gym: 2.6e-3(开2000个环境）
    # 1000: 2.2e-3
    # 2000: 2.6e-3
    # 3000: 3.0e-3
    # 4000: 3.5e-3
    # 5000: 4.0e-3
    # 10000: 7.5e-3
    data = np.asarray([[1, 1.3e-3],
                       [1000, 2.2e-3],
                       [2000, 2.6e-3],
                       [3000, 3.0e-3],
                       [4000, 3.5e-3],
                       [5000, 4.0e-3],
                       [10000, 7.5e-3]])
    plt.plot(data[:, 0], data[:, 1])
    plt.show()


if __name__ == "__main__":
    try_plot()
import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np


def main():
    with open('/home/tianchu/Documents/code_qy/puppy-gym/real_deployment/data/replay_transitions.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    replay_obs_deque = data_dict['obs_deque']
    replay_action_deque = data_dict['action_deque']
    replay_toe_pose_deque = data_dict['toe_pose_deque']

    with open('/home/tianchu/Documents/code_qy/puppy-gym/real_deployment/data/transitions.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    obs_deque = data_dict['obs_deque']
    action_deque = data_dict['action_deque']
    toe_pose_deque = data_dict['toe_pose_deque']

    # action order:
    # ['fl_j0', 'fl_j1', 'fl_j2',
    #  'fr_j0', 'fr_j1', 'fr_j2',
    #  'rl_j0', 'rl_j1', 'rl_j2',
    #  'rr_j0', 'rr_j1', 'rr_j2']

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

    plt.subplot(321)
    plt.plot(np.asarray(replay_action_deque)[:, 0], marker='.')
    plt.plot(np.asarray(replay_action_deque)[:, 1])
    plt.plot(np.asarray(replay_action_deque)[:, 2])
    plt.subplot(323)
    plt.plot(np.asarray(action_deque)[:100, 1], marker='.')
    plt.plot(np.asarray(action_deque)[:100, 2])
    # plt.plot(np.asarray(action_deque)[:100, 5])
    plt.subplot(325)
    plt.plot(np.asarray(toe_pose_deque)[:100, 1], marker='.')
    plt.plot(np.asarray(replay_toe_pose_deque)[:100, 1])
    # plt.show()

    plt.subplot(322)
    plt.plot(np.asarray(replay_action_deque)[:, 9], marker='.')
    plt.plot(np.asarray(replay_action_deque)[:, 10])
    plt.plot(np.asarray(replay_action_deque)[:, 11])
    plt.subplot(324)
    plt.plot(np.asarray(action_deque)[:100, 10], marker='.')
    plt.plot(np.asarray(action_deque)[:100, 11])
    # plt.plot(np.asarray(action_deque)[:100, 5])
    plt.subplot(326)
    plt.plot(np.asarray(toe_pose_deque)[:100, 10], marker='.')
    plt.plot(np.asarray(replay_toe_pose_deque)[:100, 10])
    plt.show()

    print('end')


if __name__ == "__main__":
    main()
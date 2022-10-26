import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np


def main():
    with open('/home/tianchu/Documents/code_qy/puppy-gym/real_deployment/data/replay_transitions.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    replay_obs_deque = data_dict['obs_deque']
    replay_action_deque = data_dict['action_deque']

    with open('/home/tianchu/Documents/code_qy/puppy-gym/real_deployment/data/transitions.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    obs_deque = data_dict['obs_deque']
    action_deque = data_dict['action_deque']
    toe_pose_deque = data_dict['toe_pose_deque']

    # plt.plot(np.asarray(action_deque)[:, 0])
    # plt.plot(np.asarray(replay_action_deque)[:, 0])
    # plt.show()

    plt.plot(np.asarray(toe_pose_deque)[:, 0])
    plt.show()
    print('end')


if __name__ == "__main__":
    main()
import copy

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCriticNet(nn.Module):
    def __init__(self,
                 observation_states,
                 observation_states_size,
                 num_actions,
                 actor_hidden_dims,
                 critic_hidden_dims,
                 activation):
        super(ActorCriticNet, self).__init__()
        self.observation_states = observation_states
        self.observation_states_size = observation_states_size

        self.copy_observation_states = copy.deepcopy(observation_states)

        self.state_encoder_dict = {}

        # =============== encoders ================ #
        if 'env_factor' in self.observation_states:
            self.copy_observation_states.remove('env_factor')
            # make the env factor encoder:
            self.env_factor_encoder = nn.Sequential(
                torch.nn.Linear(observation_states_size['env_factor'], 256),
                activation,
                torch.nn.Linear(256, 128),
                activation,
                torch.nn.Linear(128, 8),
                activation
            )
            self.state_encoder_dict['env_factor'] = self.env_factor_encoder

        if 'sequence_dof_pos' in self.observation_states:
            self.copy_observation_states.remove('sequence_dof_pos')
            # sequence observation
            self.sequence_dof_pos_encoder = nn.Sequential(
                torch.nn.Linear(observation_states_size['sequence_dof_pos'], 256),
                activation,
                torch.nn.Linear(256, 128),
                activation,
                torch.nn.Linear(128, 16),
                activation
            )
            self.state_encoder_dict['sequence_dof_pos'] = self.sequence_dof_pos_encoder

        if 'sequence_dof_action' in self.observation_states:
            self.copy_observation_states.remove('sequence_dof_action')
            # sequence observation
            self.sequence_dof_action_encoder = nn.Sequential(
                torch.nn.Linear(observation_states_size['sequence_dof_action'], 256),
                activation,
                torch.nn.Linear(256, 128),
                activation,
                torch.nn.Linear(128, 16),
                activation
            )
            self.state_encoder_dict['sequence_dof_action'] = self.sequence_dof_action_encoder

        # for other obs, will use common encoder
        self.common_encoder = None
        common_states_size = 0
        for obs in self.copy_observation_states:
            common_states_size += self.observation_states_size[obs]
        if common_states_size > 0:
            self.common_encoder = nn.Sequential(
                torch.nn.Linear(common_states_size, 256),
                activation,
                torch.nn.Linear(256, 128),
                activation,
                torch.nn.Linear(128, 16),
                activation
            )
            self.state_encoder_dict['common_encoder'] = self.common_encoder

        # =============== base policy ================ #
        policy_layers = []
        policy_layers.append(nn.Linear(48, actor_hidden_dims[0]))
        policy_layers.append(activation)

        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                policy_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                policy_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                policy_layers.append(activation)
        self.base_policy = nn.Sequential(*policy_layers)

        # =============== critic ================ #
        critic_layers = []
        critic_layers.append(nn.Linear(48, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, obs_tensor):
        X = []
        start_idx = 0
        copy_observation_states = copy.deepcopy(self.observation_states)
        only_common_states = False
        while not only_common_states and len(copy_observation_states) > 0:
            state_name = copy_observation_states.pop(0)
            if state_name in ['env_factor', 'sequence_dof_pos', 'sequence_dof_action']:
                embedding = self.state_encoder_dict[state_name](obs_tensor[..., start_idx: start_idx+self.observation_states_size[state_name]])
                X.append(embedding)
                start_idx += self.observation_states_size[state_name]
            else:
                break

        if len(copy_observation_states) > 0:
            common_states = obs_tensor[..., start_idx:]
            common_states_embedding = self.state_encoder_dict['common_encoder'](common_states)
            X.append(common_states_embedding)

        # for state in self.observation_states:
        #     if state == ''
        # if 'env_factor' in self.observation_states:
        #     # encode the env_factor:
        #     env_factor_embedding = self.env_factor_encoder(obs_dict['env_factor'])
        #     X.append(env_factor_embedding)
        # if 'sequence_dof_pos' in self.observation_states:
        #     # encode the sequence of dof_pos
        #     dof_pos_embedding = self.sequence_dof_pos_encoder(obs_dict['sequence_dof_pos'])
        #     X.append(dof_pos_embedding)
        # if 'sequence_dof_action' in self.observation_states:
        #     dof_action_embedding = self.sequence_dof_action_encoder(obs_dict['sequence_dof_action'])
        #     X.append(dof_action_embedding)
        #
        # if len(self.copy_observation_states) > 0:
        #     common_states = []
        #     for state in self.copy_observation_states:
        #         common_states.append(obs_dict[state])
        #     common_states = torch.cat(common_states, dim=-1)
        #     common_states_embedding = self.common_encoder(common_states)
        #     X.append(common_states_embedding)

        X = torch.cat(X, dim=-1)
        action_mean = self.base_policy(X)
        value = self.critic(X)
        return action_mean, value


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 observation_states=[],
                 observation_states_size={},
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        # actor_layers = []
        # actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        # actor_layers.append(activation)
        #
        # for l in range(len(actor_hidden_dims)):
        #     if l == len(actor_hidden_dims) - 1:
        #         actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
        #     else:
        #         actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
        #         actor_layers.append(activation)
        # self.actor = nn.Sequential(*actor_layers)
        # self.actor = Actor(observation_states, observation_states_size, num_actions)
        self.policy = ActorCriticNet(observation_states,
                                     observation_states_size,
                                     num_actions,
                                     actor_hidden_dims,
                                     critic_hidden_dims,
                                     activation)

        # # Value function
        # critic_layers = []
        # critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        # critic_layers.append(activation)
        # for l in range(len(critic_hidden_dims)):
        #     if l == len(critic_hidden_dims) - 1:
        #         critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
        #     else:
        #         critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
        #         critic_layers.append(activation)
        # self.critic = nn.Sequential(*critic_layers)

        # print(f"Actor MLP: {self.actor}")
        # print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean, self.value = self.policy(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean, self.value = self.policy(observations)
        self.distribution = Normal(actions_mean, actions_mean * 0. + self.std)
        return actions_mean
        # return self.distribution.sample()

    def evaluate(self, critic_observations, **kwargs):
        raise NotImplementedError
        # value = self.critic(critic_observations)
        # return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

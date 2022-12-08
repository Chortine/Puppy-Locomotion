import copy

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCriticNet(nn.Module):
    def __init__(self,
                 obs_shape_dict,
                 num_actions,
                 actor_hidden_dims,
                 critic_hidden_dims,
                 activation,
                 regression_loss
                 ):

        super(ActorCriticNet, self).__init__()
        self.obs_groups = list(obs_shape_dict.keys())
        self.states_in_regular_obs = list(obs_shape_dict['obs_tensor'].keys())
        self.regular_obs_shape_dict = obs_shape_dict['obs_tensor']
        self.state_encoder_dict = {}

        self.policy_input_size = 0
        self.compute_regression_loss = regression_loss

        # ============== rma obs mem CNN encoder =========== #
        if 'rma_obs_mem' in self.obs_groups:
            # capture the temporal relations
            rma_obs_size = obs_shape_dict['rma_obs_mem'][0]
            self.adaptation_module_encoder = nn.Sequential(
                nn.Conv1d(in_channels=rma_obs_size, out_channels=rma_obs_size, kernel_size=8, stride=4,
                          padding=0),
                activation,
                nn.Conv1d(in_channels=rma_obs_size, out_channels=rma_obs_size, kernel_size=5, stride=1,
                          padding=0),
                activation,
                nn.Conv1d(in_channels=rma_obs_size, out_channels=rma_obs_size, kernel_size=5, stride=1,
                          padding=0),
                activation,
                nn.Flatten(),
                nn.Linear(66, 8),
                activation
            )
            # self.adaptation_module_encoder.append(nn.Flatten())
            # self.adaptation_module_encoder.append(nn.Linear(66, 8))
            # self.adaptation_module_encoder.append(activation)

            self.z_loss = nn.MSELoss()

        # =============== encoders ================ #
        if 'env_factor' in self.states_in_regular_obs:
            embedding_size = 8
            # make the env factor encoder:
            self.env_factor_encoder = nn.Sequential(
                torch.nn.Linear(self.regular_obs_shape_dict['env_factor'], 256),
                activation,
                torch.nn.Linear(256, 128),
                activation,
                torch.nn.Linear(128, embedding_size),
                activation
            )
            self.state_encoder_dict['env_factor'] = self.env_factor_encoder
            self.policy_input_size += embedding_size

        if 'sequence_dof_pos' in self.states_in_regular_obs:
            embedding_size = 16
            # sequence observation
            self.sequence_dof_pos_encoder = nn.Sequential(
                torch.nn.Linear(self.regular_obs_shape_dict['sequence_dof_pos'], 256),
                activation,
                torch.nn.Linear(256, 128),
                activation,
                torch.nn.Linear(128, embedding_size),
                activation
            )
            self.state_encoder_dict['sequence_dof_pos'] = self.sequence_dof_pos_encoder
            self.policy_input_size += embedding_size

        if 'sequence_dof_action' in self.states_in_regular_obs:
            embedding_size = 16
            # sequence observation
            self.sequence_dof_action_encoder = nn.Sequential(
                torch.nn.Linear(self.regular_obs_shape_dict['sequence_dof_action'], 256),
                activation,
                torch.nn.Linear(256, 128),
                activation,
                torch.nn.Linear(128, embedding_size),
                activation
            )
            self.state_encoder_dict['sequence_dof_action'] = self.sequence_dof_action_encoder
            self.policy_input_size += embedding_size

        # for other obs, will use common encoder
        if 'common_states' in self.states_in_regular_obs:
            embedding_size = 64
            common_states_size = sum(list(self.regular_obs_shape_dict['common_states'].values()))
            self.common_encoder = nn.Sequential(
                torch.nn.Linear(common_states_size, 256),
                activation,
                torch.nn.Linear(256, 128),
                activation,
                torch.nn.Linear(128, embedding_size),
                activation
            )
            self.state_encoder_dict['common_states'] = self.common_encoder
            self.policy_input_size += embedding_size

        # =============== base policy ================ #
        policy_layers = []
        policy_layers.append(nn.Linear(self.policy_input_size, actor_hidden_dims[0]))
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
        critic_layers.append(nn.Linear(self.policy_input_size, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, obs_dict: dict):
        if 'rma_obs_mem' in self.obs_groups:
            # encode rma_obs_mem
            rma_obs_mem = obs_dict['rma_obs_mem']
            z_rma_mem = self.adaptation_module_encoder(rma_obs_mem)

        obs_tensor = obs_dict['obs_tensor']
        X = []
        start_idx = 0
        for state_name in self.states_in_regular_obs:
            if state_name == 'common_states':
                state_size = sum(list(self.regular_obs_shape_dict['common_states'].values()))
            else:
                state_size = self.regular_obs_shape_dict[state_name]

            states_embedding = self.state_encoder_dict[state_name](obs_tensor[..., start_idx: start_idx + state_size])
            if state_name == 'env_factor':
                z_env_factor = states_embedding
                if 'rma_obs_mem' in self.obs_groups:
                    X.append(z_rma_mem)
                else:
                    X.append(z_env_factor)
            else:
                X.append(states_embedding)

            start_idx += state_size

        X = torch.cat(X, dim=-1)
        action_mean = self.base_policy(X)
        value = self.critic(X)
        if self.compute_regression_loss:
            # num_envs * 8
            z_loss = self.z_loss(z_rma_mem, z_env_factor)
            return action_mean, value, z_loss
        else:
            return action_mean, value


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self,
                 obs_shape_dict,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()
        self.obs_shape_dict = obs_shape_dict
        if 'rma_obs_mem' in self.obs_shape_dict.keys():
            self.rma_regression_loss = True
        else:
            self.rma_regression_loss = False
        # FIXME: use z_rma_mem but not compute regression loss
        # self.rma_regression_loss = False
        activation = get_activation(activation)

        # Policy
        self.policy = ActorCriticNet(obs_shape_dict,
                                     num_actions,
                                     actor_hidden_dims,
                                     critic_hidden_dims,
                                     activation,
                                     self.rma_regression_loss
                                     )
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        if self.rma_regression_loss:
            self.freeze_parameters()
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def freeze_parameters(self):
        for name, param in self.named_parameters():
            if 'policy.adaptation_module_encoder' not in name:
                param.requires_grad = False

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
        if self.rma_regression_loss:
            mean, self.value, self.z_loss = self.policy(observations)
        else:
            mean, self.value = self.policy(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    @property
    def rma_z_loss(self):
        return self.z_loss

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        if self.rma_regression_loss:
            actions_mean, self.value, self.z_loss = self.policy(observations)
        else:
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

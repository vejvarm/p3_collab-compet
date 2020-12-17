# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class DDPGAgent:
    def __init__(self,
                 in_actor,
                 out_actor,
                 n_filt_actor,
                 kernel_size_actor,
                 stride_actor,
                 fc_units_actor,
                 in_critic,
                 n_filt_critic,
                 kernel_size_critic,
                 stride_critic,
                 fc_units_critic,
                 lr_actor=1.0e-3,
                 lr_critic=1.0e-5):  # 1e-5 was getting to 0.4 score (sporadically)
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, out_actor, n_filt_actor, kernel_size_actor, stride_actor, fc_units_actor, actor=True).to(device)
        self.critic = Network(in_critic, 1, n_filt_critic, kernel_size_critic, stride_critic, fc_units_critic).to(device)
        self.target_actor = Network(in_actor, out_actor, n_filt_actor, kernel_size_actor, stride_actor, fc_units_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, 1, n_filt_critic, kernel_size_critic, stride_critic, fc_units_critic).to(device)

        self.noise = OUNoise(out_actor, scale=.1)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-3)

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs) + noise*self.noise.noise()
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action

import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F


class ContextBandit:
    def __init__(self, states=10, arms=10):
        self.state = None
        self.bandit_matrix = None
        self.arms = arms
        self.states = states
        self.init_distribution(states, arms)
        self.update_state()

    def init_distribution(self, states, arms):
        self.bandit_matrix = np.random.rand(states, arms)

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        return self.state

    def update_state(self):
        self.state = np.random.randint(0, self.states)

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward


class ContextBanditModel(nn.Module):
    def __init__(self, states=10, arms=10, hidden=100):
        super(ContextBanditModel, self).__init__()
        self.states = states
        self.arms = arms
        self.hidden = hidden
        self.l1 = nn.Linear(arms, hidden)
        self.l2 = nn.Linear(hidden, arms)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))

        return out


class MiniGridModel(nn.Module):
    def __init__(self, l1_input=64, l1_output=150, l2_output=100, output=4):
        super(MiniGridModel, self).__init__()
        self.l1_input = l1_input
        self.l1_output = l1_output
        self.l2_output = l2_output
        self.output = output

        self.l1 = nn.Linear(l1_input, l1_output)
        self.l2 = nn.Linear(l1_output, l2_output)
        self.l3 = nn.Linear(l2_output, output)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out


class Reinforce(nn.Module):
    def __init__(self):
        super(Reinforce, self).__init__()

        self.agent = nn.Sequential(
            nn.Linear(4, 150),
            nn.LeakyReLU(),
            nn.Linear(150, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        out = self.agent(x)
        return out


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0)

        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic


class Genetic(nn.Module):
    def __init__(self):
        super(Genetic, self).__init__()

        self.agent = nn.Sequential(
            nn.Linear(4, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        out = self.agent(x)
        out = F.log_softmax(out, dim=0)

        return out
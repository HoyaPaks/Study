import gym
import numpy as np
import torch
from torch.nn import functional as F
from torch import optim
from IPython.display import clear_output


def run_episode_monte(worker_env, worker_model):
    """
    Run this function once per epochs to collect data
    Monte-Carlo method (Episodic)

    worker_env : Environment
    worker_model : Agent
    """
    state = torch.from_numpy(worker_env.env.state).float()
    values, logprobs, rewards = [], [], []
    done = False

    while not done:
        policy, value = worker_model(state)

        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()

        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, reward, done, truncated = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()

        if done:
            reward = -10
            worker_env.reset()

        else:
            reward = 1.0
        rewards.append(reward)

    return values, logprobs, rewards


def update_params_monte(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)  # A
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)

    Returns = []
    ret_ = torch.Tensor([0])

    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)

    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)

    actor_loss = -1 * logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns, 2)

    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)


def worker_monte(worker_model, epochs):
    score = []

    worker_env = gym.make("CartPole-v1", render_mode="rgb_array")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()

    for i in range(epochs):
        print(f"epochs : {i}")
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode_monte(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params_monte(worker_opt, values, logprobs, rewards)
        score.append(eplen + 10)
        clear_output(wait=True)

    return score


def run_episode_n_step(worker_env, worker_model, N_steps=100):
    """
    Run this function every N_steps in an epoch to collect data
    N_step method (Online - N steps - Monte Carlo)

    worker_env : Environment
    worker_model : Agent
    N_steps : Hyper-parameter for how many times updating loss in an epoch
    """

    raw_state = np.array(worker_env.env.state)

    state = torch.from_numpy(raw_state).float()
    values, logprobs, rewards = [], [], []
    done = False

    j = 0
    G = torch.Tensor([0])

    while j < N_steps and done == False:
        j += 1
        policy, value = worker_model(state)
        values.append(value)

        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()

        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)

        state_, reward, done, truncated = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()

        if done:
            reward = -10
            worker_env.reset()
            G = torch.Tensor([0])

        else:
            reward = 1.0
            G = value.detach()
        rewards.append(reward)

    return values, logprobs, rewards, G


def update_params_n_step(worker_opt, values, logprobs, rewards, G, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)  # A
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)

    Returns = []
    # print(G)
    ret_ = G

    for r in range(rewards.shape[0]):  # B
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)

    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)

    actor_loss = -1 * logprobs * (Returns - values.detach())  # C
    critic_loss = torch.pow(values - Returns, 2)  # D

    loss = actor_loss.sum() + clc * critic_loss.sum()  # E
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)


def worker_n_step(worker_model, epochs):
    score = []

    worker_env = gym.make("CartPole-v1", render_mode="rgb_array")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()

    for i in range(epochs):
        print(f"epochs : {i}")
        worker_opt.zero_grad()
        values, logprobs, rewards, G = run_episode_n_step(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params_n_step(worker_opt, values, logprobs, rewards, G)
        score.append(eplen + 10)
        clear_output(wait=True)

    return score
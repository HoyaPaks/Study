import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


def genetic_model(x, unpacked_params):
    l1, b1, l2, b2, l3, b3 = unpacked_params
    y = F.linear(x, l1, b1)  # B
    y = torch.relu(y)  # C
    y = F.linear(y, l2, b2)
    y = torch.relu(y)
    y = F.linear(y, l3, b3)
    y = torch.log_softmax(y, dim=0)  # D
    return y


def unpack_params(params, layers=[(25, 4), (10, 25), (2, 10)]):
    unpacked_params = []
    end = 0

    for index, layer in enumerate(layers):
        start, end = end, end + np.prod(layer)
        weights = params[start:end].view(layer)

        start, end = end, end + layer[0]
        bias = params[start:end]
        unpacked_params.extend([weights, bias])

    return unpacked_params


def spawn_population(N=50, size=407):
    population = []
    for i in range(N):
        random_vec = torch.randn(size) / 2.0
        fit = 0
        p = {'params': random_vec, 'fitness': fit}
        population.append(p)

    return population


def recombine(x1, x2):
    x1 = x1['params']
    x2 = x2['params']
    length = x1.shape[0]
    split_pt = np.random.randint(length)

    child1 = torch.zeros(length)
    child2 = torch.zeros(length)
    child1[0:split_pt] = x1[0:split_pt]
    child1[split_pt:] = x2[split_pt:]
    child2[0:split_pt] = x2[0:split_pt]
    child2[split_pt:] = x1[split_pt:]

    c1 = {'params': child1, 'fitness': 0.0}
    c2 = {'params': child2, 'fitness': 0.0}
    return c1, c2


def mutate(x, rate=0.01):
    x_ = x['params']
    num_to_change = int(rate * x_.shape[0])
    idx = np.random.randint(low=0, high=x_.shape[0], size=(num_to_change,))
    x_[idx] = torch.randn(num_to_change) / 10.0
    x['params'] = x_
    return x


def test_model(env, model, agent):
    done = False
    state_ = env.reset()
    state = torch.from_numpy(state_).float()
    score = 0

    while not done:
        params = unpack_params(agent['params'])
        probs = model(state, params)
        action = torch.distributions.Categorical(probs=probs).sample()
        state_, reward, done, _ = env.step(action.item())
        state = torch.from_numpy(state_).float()
        score += 1
    return score


def evaluate_population(population, env, model):
    total_fit = 0
    population_size = len(population)
    for agent in population:
        score = test_model(env, model, agent)
        agent['fitness'] = score
        total_fit += score
    avg_fit = total_fit / population_size
    return population, avg_fit


def next_generation(pop, mut_rate=0.001, tournament_size=0.2):
    new_pop = []
    lp = len(pop)
    while len(new_pop) < len(pop):
        random_idx = np.random.randint(low=0, high=lp, size=(int(tournament_size * lp)))
        batch = np.array([[index, params['fitness']] for (index, params) in enumerate(pop) if index in random_idx])
        scores = batch[batch[:, 1].argsort()]
        idx_0, idx_1 = int(scores[-1][0]), int(scores[-2][0])

        parent0, parent1 = pop[idx_0], pop[idx_1]
        offspring0, offspring1 = recombine(parent0, parent1)
        child1 = mutate(offspring0, rate=mut_rate)
        child2 = mutate(offspring1, rate=mut_rate)
        offspring = [child1, child2]
        new_pop.extend(offspring)

    return new_pop

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.distributions.categorical import Categorical
def plot_reward(rewards,name,algo):
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)
    x = np.linspace(0, len(rewards), len(rewards))
    ax.plot(
            x,
            rewards,
            color='k')
    ax.set_title(algo)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avergage cumulative reward')
    ax.legend(loc='best')
    plt.savefig(name)


# select - actions
def select_actions(pi, deterministic=False):
    cate_dist = Categorical(pi)
    if deterministic:
        return torch.argmax(pi, dim=1).item()
    else:
        return cate_dist.sample().unsqueeze(-1)

# get the action log prob and entropy...
def evaluate_actions(pi, actions):
    cate_dist = Categorical(pi)
    return cate_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1), cate_dist.entropy().mean()

# get the action log prob and entropy...
def evaluate_actions_sil(pi, actions):
    cate_dist = Categorical(pi)
    return cate_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1), cate_dist.entropy().unsqueeze(-1)

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.-done)
        discounted.append(r)
    return discounted[::-1]
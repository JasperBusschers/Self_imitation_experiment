import matplotlib.pyplot as plt
import numpy as np

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
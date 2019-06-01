import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from torch import optim

from a2c_module import Memory, a2c
from sil_module import sil_module

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    ############## Hyperparameters ##############
    env_name = "LunarLanderContinuous-v2"
    render = False
    solved_reward = 200  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 50000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.6  # constant std for action distribution
    lr = 0.0025
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 1  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    SIL = False
    ##########SIL hyperparameters################
    N_updates = 5
    max_logp = 0.1
    max_grad_norm = 0.5
    batch_size = 512
    capacity = 1000
    sil_alpha= 0.6
    sil_beta =0.1
    clip_par =1
    w_value = 0.01
    #############################################
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    pol = a2c(state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        states = []
        actions = []
        rewards = []
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action ,tensor_state , tensor_action= pol.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            pol.sil_model.step(state,action,reward,done)
            states.append(tensor_state.detach())
            actions.append(tensor_action.detach())
            rewards.append(reward)
            # Saving reward:
            memory.rewards.append(reward)
            # update if its time
            if time_step % update_timestep == 0:


                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                pol.update(memory)
                if SIL:
                    pol.update_off_policy()
                memory.clear_memory()
                break
        avg_length += t

        # # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(pol.policy.state_dict(), './PPO_Continuous_{}.pth'.format(env_name))
            break
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
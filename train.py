import torch
import gym
import numpy as np
from a2c_module import a2c
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args):
    # creating environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if args.random_seed:
        print("Random Seed: {}".format(args.random_seed))
        torch.manual_seed(args.random_seed)
        env.seed(args.random_seed)
        np.random.seed(args.random_seed)

    pol = a2c(state_dim,action_dim,args)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, args.max_episodes + 1):
        state = env.reset()
        done = False
        while not done:
            time_step += 1
            # Running policy_old:
            action = pol.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # Saving reward:
            pol.policy.rewards.append(reward)
            state = next_state
            running_reward += reward
            if args.render:
                env.render()
            if done:
                break
        pol.update()
        if args.SIL:
            pol.update_off_policy()
        # # stop training if avg_reward > solved_reward
        if running_reward > (args.log_interval * args.solved_reward):
            print("########## Solved! ##########")
            torch.save(pol.policy.state_dict(), './PPO_Continuous_{}.pth'.format(args.env_name))
            break
        # logging
        if i_episode % args.log_interval == 0:
            avg_length = int(time_step /i_episode)
            running_reward = int((running_reward / args.log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0


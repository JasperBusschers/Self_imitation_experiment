from train import train
import argparse

def arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument('--env_name', type=str, default='LunarLander-v2', help='the environment name')
    parse.add_argument('--continious', default=False, action='store_false', help='continious environment')
    parse.add_argument('--solved_reward', type=int, default=200, help='reward for solving environment')
    parse.add_argument('--log_interval', type=int, default=20,help='log every n episodes')
    parse.add_argument('--max_episodes', type=int, default=50000,help='maximum number of episodes')
    parse.add_argument('--n_latent_var', type=int, default=64, help='dimension hidden layer')
    parse.add_argument('--action_std', type=float, default=0.6, help='standard deviation actions')
    parse.add_argument('--lr', type=float, default=0.0025, help='learning rate')

    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--random_seed', type=int, default=123, help='random seed for numpy and pytorch')
    parse.add_argument('--K_epochs', type=int, default=1, help='number of updates policy')
    parse.add_argument('--K_epochs_sil', type=int, default=1, help='number of updates sil')
    parse.add_argument('--SIL',default=False, action='store_false', help='check if use the sil')
    parse.add_argument('--render', default=False,action='store_false', help='check if use the sil')

    # SIL PARAMETERS
    parse.add_argument('--batch_size', type=int, default=512, help='the batch size to update the sil module')
    parse.add_argument('--capacity', type=int, default=50, help='buffer capacity')
    parse.add_argument('--mini-batch-size', type=int, default=64, help='the minimal batch size')
    parse.add_argument('--clip', type=float, default=1, help='clip parameters')
    parse.add_argument('--entropy-coef', type=float, default=0.01, help='entropy-reg')
    parse.add_argument('--w-value', type=float, default=0.01, help='the wloss coefficient')
    parse.add_argument('--max-grad-norm', type=float, default=0.5, help='the grad clip')

    args = parse.parse_args()

    return args

train(arguments())
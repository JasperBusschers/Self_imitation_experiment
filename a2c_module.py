import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

from discriminator_module import discriminator_module
from sil_module import sil_module
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,args):
        super(ActorCritic, self).__init__()
        n_var = args.n_latent_var
        self.continious = args.continious
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.Tanh(),
            nn.Linear(n_var, n_var),
            nn.Tanh(),
            nn.Linear(n_var, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.Tanh(),
            nn.Linear(n_var, n_var),
            nn.Tanh(),
            nn.Linear(n_var, 1)
        )
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.actions = []
        self.states = []
        if args.continious:
            self.action_var = torch.full((action_dim,), args.action_std * args.action_std).to(device)

    def forward(self, state):
        self.states.append(state[0])
        state = torch.from_numpy(state).float().to(device)
        state_value = self.critic(state)
        action_feats = self.actor(state)
        if self.continious:
            action = self.forward_continious(state_value,action_feats)
        else:
            action=  self.forward_discrete(state_value,action_feats)
        self.actions.append(action)
        return action

    def forward_discrete(self, state_value,action_feats):
        action_probs = F.softmax(action_feats, dim=1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        return action.item()

    def forward_continious(self, state_value,action_feats):
        action_distribution = MultivariateNormal(action_feats, torch.diag(self.action_var).to(device))
        action = action_distribution.sample()
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        return action.cpu().data.numpy().flatten()

    def evaluate(self,state, action):
        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().to(device)
        state_value = self.critic(state)
        action_feats = self.actor(state)
        if self.continious:
            dist = MultivariateNormal(torch.squeeze(action_feats), torch.diag(self.action_var))
            action_logprobs = dist.log_prob(torch.squeeze(action))
            dist_entropy = dist.entropy()
        else:
            action_probs = F.softmax(action_feats, dim=1)
            dist  = Categorical(action_probs)
            action_logprobs = dist.log_prob(torch.squeeze(action))
            dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def calculateLoss(self, rewards):
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value[0][0], reward)
            loss += (action_loss + value_loss)
        return loss

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.states[:]
        del self.actions[:]

class a2c:
    def __init__(self,state_dim,action_dim,args):
        self.args = args
        self.policy =  ActorCritic(state_dim,action_dim,args).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=args.lr, betas=(0.9,0.999))
        self.sil_model = sil_module(self.policy,self.optimizer,args)
        if args.disc:
            self.discriminator = discriminator_module(state_dim,action_dim,args)

    def select_action(self, state):
        return self.policy(np.asarray([state]))

    def update(self):
        returns ,dis_reward= self.calc_discounted_reward()
        for _ in range(self.args.K_epochs):
            self.optimizer.zero_grad()
            loss = self.policy.calculateLoss(returns)
            loss.backward()
            self.optimizer.step()
        sample = {'states': np.asarray(self.policy.states),
                  'actions': np.asarray(self.policy.actions),
                  'rewards': returns.cpu().numpy()}#self.policy.rewards)}
        #if self.args.SIL or self.args.disc:
        #    self.sil_model.good_buffer.add(sample,dis_reward)
        if self.args.disc:
            loss_disc =self.discriminator.train_discriminator(self.policy.states,self.policy.actions, self.sil_model.good_buffer)
            print(loss_disc)
        self.policy.clearMemory()

    def update_off_policy(self):
        return  self.sil_model.train_sil_model()

    def calc_discounted_reward(self):
        # calculating discounted rewards:
        if self.args.disc:
            disc_rewards = self.discriminator.reward_dis(self.policy.states, self.policy.actions).detach()
        else:
            disc_rewards = [0 for _ in self.policy.rewards[::-1]]
            disc_rewards = torch.Tensor(disc_rewards).to(device)
        rewards_discriminator = []
        disc_rew = 0
        rewards = []
        discounted_reward = 0
        for reward, disc_r in zip(self.policy.rewards[::-1], disc_rewards):
            disc_rew = disc_r+ self.args.gamma * disc_rew
            discounted_reward =reward + self.args.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
            rewards_discriminator.insert(0, disc_rew)
        # normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards_discriminator = torch.tensor(rewards_discriminator).to(device)
        rewards =  self.args.weight_environment_reward * ( (rewards - rewards.mean()) / (rewards.std())) + self.args.weight_disc *disc_rewards# ( (rewards_discriminator - rewards_discriminator.mean()))    #( (rewards - rewards.mean()) / (rewards.std()))
        return rewards, discounted_reward

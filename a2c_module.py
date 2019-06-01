import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from buffer import PriorityQueueSet
from sil_module import sil_module
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,args):
        super(ActorCritic, self).__init__()
        n_var = args.n_latent_var
        self.continious = args.continious
        if args.continious:
            self.action_var = torch.full((action_dim,), args.action_std * args.action_std).to(device)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.Tanh(),
            nn.Linear(n_var, n_var),
            nn.Tanh(),
            nn.Linear(n_var, action_dim),
            nn.Tanh()
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



    def forward(self, state):
        self.states.append(state[0])
        state = torch.from_numpy(state).float().to(device)
        if self.continious:
            action = self.forward_continious(state)
        else:
            action=  self.forward_discrete(state)
        self.actions.append(action)
        return action

    def forward_discrete(self, state):
        state_value = self.critic(state)
        action_feats = self.actor(state)
        action_probs = F.softmax(action_feats, dim=0)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        return action.item()

    def forward_continious(self, state):
        state_value = self.critic(state)
        action_feats = self.actor(state)
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
            action_probs = F.softmax(action_feats, dim=0)
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
                  'rewards': np.asarray(self.policy.rewards)}
        if self.args.SIL:
            self.sil_model.good_buffer.add(sample,dis_reward)
        self.policy.clearMemory()


    def update_off_policy(self):
        return  self.sil_model.train_sil_model()


    def calc_discounted_reward(self):
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.policy.rewards[::-1]:
            dis_reward = reward + self.args.gamma * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        return rewards, dis_reward

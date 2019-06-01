import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from sil_module import sil_module
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.Tanh(),
            nn.Linear(n_var, n_var),
            nn.Tanh(),
            nn.Linear(n_var, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.Tanh(),
            nn.Linear(n_var, n_var),
            nn.Tanh(),
            nn.Linear(n_var, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var).to(device))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        dist = MultivariateNormal(torch.squeeze(action_mean), torch.diag(self.action_var))

        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class a2c:
    def __init__(self, state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip, TRPO = False):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.TRPO = TRPO
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        self.MseLoss = nn.MSELoss()
        self.sil_model = sil_module(self.policy,self.optimizer)


    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if self.TRPO:
            action = self.policy_old.act(state, memory).cpu()
        else:
            action = self.policy.act(state, memory).cpu()
        return action.data.numpy().flatten(), state, action

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        #for reward in reversed(memory.rewards):
        #    discounted_reward = reward + (self.gamma * discounted_reward)
        #    rewards.insert(0, discounted_reward)
        R=0
        for r in memory.rewards:
            R = r + self.gamma * R
            rewards.insert(0, R)
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            advantages = rewards - state_values.detach()
            if self.TRPO:
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())
                # Finding Surrogate Loss:
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2)
            else:
                policy_loss = -logprobs * advantages
            loss = policy_loss + 0.5 * F.smooth_l1_loss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Copy new weights into old policy:
        if self.TRPO:
            self.policy_old.load_state_dict(self.policy.state_dict())
    def update_off_policy(self):
        return  self.sil_model.train_sil_model()

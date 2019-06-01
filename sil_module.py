import numpy as np
import torch
from memory import PrioritizedReplayBuffer


class sil_module:
    def __init__(self, network, optimizer):
        self.n_update = 5
        self.network = network
        self.running_episodes = []
        self.optimizer = optimizer
        self.buffer = PrioritizedReplayBuffer(10, 0.5)
        # some other parameters...
        self.total_steps = []
        self.total_rewards = []

    # add the batch information into it...
    def step(self, obs, actions, rewards, dones):
        self.running_episodes.append([obs, actions, rewards])
        # to see if can update the episode...
        if dones:
                self.update_buffer(self.running_episodes)
                self.running_episodes = []
    
    # train the sil model...
    def train_sil_model(self):
        for n in range(self.n_update):
            obs, actions, returns, weights, idxes = self.sample_batch(64)
            mean_adv, num_valid_samples = 0, 0
            if obs is not None:
                # need to get the masks
                # get basic information of network..
                obs = torch.tensor(obs, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
                returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
                max_nlogp = torch.tensor(np.ones((len(idxes), 1)) * 0.5, dtype=torch.float32)
                if True:# self.args.cuda:
                    obs = obs.cuda()
                    actions = actions.cuda()
                    returns = returns.cuda()
                    weights = weights.cuda()
                    max_nlogp = max_nlogp.cuda()
                # start to next...
                action_log_probs, value, dist_entropy = self.network.evaluate(obs,actions)
                #action_log_probs, dist_entropy = evaluate_actions_sil(pi, actions)
                action_log_probs = -action_log_probs
                clipped_nlogp = torch.min(action_log_probs, max_nlogp)
                # process returns
                advantages = returns - value
                advantages = advantages.detach()
                masks = (advantages.cpu().numpy() > 0).astype(np.float32)
                # get the num of vaild samples
                num_valid_samples = np.sum(masks)
                num_samples = np.max([num_valid_samples, 64])
                # process the mask
                masks = torch.tensor(masks, dtype=torch.float32)
                if True:#self.args.cuda:
                    masks = masks.cuda()
                # clip the advantages...
                clipped_advantages = torch.clamp(advantages, 0, 0.5)
                mean_adv = torch.sum(clipped_advantages) / num_samples 
                mean_adv = mean_adv.item() 
                # start to get the action loss...
                action_loss = torch.sum(clipped_advantages * weights * clipped_nlogp) / num_samples
                entropy_reg = torch.sum(weights * dist_entropy * masks) / num_samples
                policy_loss = action_loss - entropy_reg * 0.01
                # start to process the value loss..
                # get the value loss
                delta = torch.clamp(value - returns, -0.5, 0) * masks
                delta = delta.detach()
                value_loss = torch.sum(weights * value * delta) / num_samples
                total_loss = policy_loss + 0.5 * 0.5 * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        return mean_adv, num_valid_samples
    
    # update buffer
    def update_buffer(self, trajectory):
        positive_reward = False
        for (ob, a, r) in trajectory:
            if r > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.total_steps) > 10 and len(self.total_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        obs = []
        actions = []
        rewards = []
        dones = []
        for (ob, action, reward) in trajectory:
            if ob is not None:
                obs.append(ob)
            else:
                obs.append(None)
            actions.append(action)
            rewards.append(np.sign(reward))
            dones.append(False)
        dones[len(dones) - 1] = True
        returns = self.discount_with_dones(rewards, dones, 0.999)
        for (ob, action, R) in list(zip(obs, actions, returns)):
            self.buffer.add(ob, action, R)

    def fn_reward(self, reward):
        return np.sign(reward)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0
    
    def num_episodes(self):
        return len(self.total_rewards)

    def num_steps(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=0.5)
        else:
            return None, None, None, None, None

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]

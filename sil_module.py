import numpy as np
import torch
from buffer import PriorityQueueSet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class sil_module:
    def __init__(self, network, optimizer,args):
        self.args = args
        self.network = network
        self.running_episodes = []
        self.optimizer = optimizer
        # some other parameters...
        self.total_steps = []
        self.total_rewards = []
        self.good_buffer = PriorityQueueSet(args.capacity)


    # train the sil model...
    def train_sil_model(self):
        for n in range(self.args.K_epochs_sil):
            best_sample = self.good_buffer.sample()
            obs = np.asarray(best_sample['states'])
            actions = np.asarray(best_sample['actions'])
            returns =  best_sample['rewards']
            mean_adv, num_valid_samples = 0, 0
            if obs is not None:
                returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(device)
                max_nlogp = torch.tensor(np.ones((len(actions), 1)) * 0.5, dtype=torch.float32).to(device)
                # start to next...
                action_log_probs, value, dist_entropy = self.network.evaluate(obs,actions)
                action_log_probs = -action_log_probs
                clipped_nlogp = torch.min(action_log_probs, max_nlogp)
                # process returns
                advantages = returns - value
                advantages = advantages.detach()
                masks = (advantages.cpu().numpy() > 0).astype(np.float32)
                # get the num of vaild samples
                num_valid_samples = np.sum(masks)
                num_samples = np.max([num_valid_samples, self.args.mini_batch_size])
                # process the mask
                masks = torch.tensor(masks, dtype=torch.float32).to(device)
                # clip the advantages...
                clipped_advantages = torch.clamp(advantages, 0,self.args.clip)
                mean_adv = torch.sum(clipped_advantages) / num_samples 
                mean_adv = mean_adv.item() 
                # start to get the action loss...
                weights=1
                action_loss = torch.sum(clipped_advantages * weights * clipped_nlogp) / num_samples
                entropy_reg = torch.sum(weights * dist_entropy * masks) / num_samples
                policy_loss = action_loss - entropy_reg * self.args.entropy_coef
                # start to process the value loss..
                # get the value loss
                delta = torch.clamp(value - returns, -0.5, 0) * masks
                delta = delta.detach()
                value_loss = torch.sum(weights * value * delta) / num_samples
                total_loss = policy_loss + 0.5 * self.args.w_value * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),  self.args.max_grad_norm)
                self.optimizer.step()
        return mean_adv, num_valid_samples
    


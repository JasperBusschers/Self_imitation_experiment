import torch
import torch.nn as nn
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super().__init__()
        self.args = args
        if  args.continious:
            self.action_dim =2
        else:
            self.action_dim =action_dim
        n_latent_var = args.n_latent_var
        self.fc = nn.Sequential(
            nn.Linear(state_dim +  self.action_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1),
            nn.Sigmoid()
        )

    def forward(self, obs, actions):
        if not self.args.continious:
            actions=actions.unsqueeze(1)
            actions_onehot = torch.zeros((obs.shape[0], self.action_dim)).to(device).float()
            actions_onehot.zero_()
            actions_onehot.scatter_(1, actions.long(), 1)
            actions = actions_onehot
        h = torch.cat((actions.float(), obs.float()), 1)
        h = self.fc(h)
        return h


class discriminator_module:
    def __init__(self, state_dim, action_dim,args):
        self.args = args
        self.discriminator = Discriminator(state_dim, action_dim,args).to(device)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.9,0.999))
        self.criterion = nn.BCELoss(reduction='none')


    def reward_dis(self, s, a):
        s,a = torch.tensor(s).to(device),torch.tensor(a).to(device)
        d_reward = self.discriminator(s, a)#+ torch.log(1 - self.discriminator(s, a))
        return d_reward

    def train_discriminator(self, states,actions,buffer):
        if buffer.counter != 0:
            for best_samples in buffer.get_batches():
                sampled_obs = torch.tensor(states).to(device).detach()
                sampled_action = torch.tensor(actions).to(device).detach()
                best_sampled_obs =torch.tensor(best_samples['states']).to(device).detach()
                bes_sampled_action = torch.tensor(best_samples['actions']).to(device).detach()
                self.optimizerD.zero_grad()
                b_size = best_sampled_obs.size(0)
                label = torch.full((b_size,), 1, device=device)
                output = self.discriminator(best_sampled_obs,bes_sampled_action).view(-1)
                errD_real = self.criterion(output, label)
                b_size = sampled_obs.size(0)
                label = torch.full((b_size,), 0, device=device)
                output = self.discriminator(sampled_obs, sampled_action).view(-1)
                errD_fake = self.criterion(output, label)
                lossD = torch.mean(errD_real) + torch.mean(errD_fake)
                lossD.backward()
                self.optimizerD.step()
                loss = errD_fake.mean().item()
        return loss



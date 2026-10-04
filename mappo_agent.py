import torch
import torch.nn as nn
from torch.distributions import Normal

N_MAX = 5
G_MAX = 500
P_MAX = 300

class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
            
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
        
        self.action_logstd = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs):
        return self.net(obs)
        
    def get_action_and_logprob(self, obs, action=None):
        mu = self.forward(obs)
        std = self.action_logstd.exp().expand_as(mu)
        
        dist = Normal(mu, std)
        
        if action is None:
            action = dist.sample()
            
        logprob = dist.log_prob(action).sum(axis=-1)
        return action, logprob

class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim, N=N_MAX, hidden_sizes=(64, 64)):
        super().__init__()
        self.N = N
        self.obs_dim = obs_dim
        
        in_dim = N * obs_dim + N
        
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
            
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, global_obs, agent_id):
        """
        global_obs: (batch_size, N * obs_dim)
        agent_id: (batch_size,) long tensor
        """
        batch_size = global_obs.shape[0]
        
        one_hot = torch.zeros((batch_size, self.N), device=global_obs.device)
        one_hot[torch.arange(batch_size), agent_id] = 1.0
        
        x = torch.cat([global_obs, one_hot], dim=-1)
        value = self.net(x)
        return value.squeeze(-1)

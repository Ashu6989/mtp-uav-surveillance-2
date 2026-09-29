import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mappo_agent import SharedActor, CentralizedCritic
from train_mappo import compute_gae, compute_losses
from torch.distributions import Normal

def test_log_prob_matches():
    mu = torch.tensor([0.5, -0.2, 0.1])
    std = torch.tensor([1.0, 0.5, 2.0])
    a = torch.tensor([0.4, 0.0, -0.5])
    
    dist = Normal(mu, std)
    expected_log_prob = dist.log_prob(a).sum(-1)
    
    actor = SharedActor(10, 3)
    # mock forward
    actor.forward = lambda obs: mu
    actor.action_logstd.data = torch.log(std)
    
    _, log_prob = actor.get_action_and_logprob(torch.zeros(10), a)
    
    assert torch.abs(log_prob - expected_log_prob) < 1e-5

def test_critic_actor_dims_and_perturbation():
    N = 3
    obs_dim = 24
    
    actor = SharedActor(obs_dim, 3)
    critic = CentralizedCritic(obs_dim, N)
    
    assert critic.net[0].in_features == 3 * 24 + 3 # 75
    assert actor.net[0].in_features == 24
    
    # Perturbation
    obs1 = torch.rand(24)
    obs2 = torch.rand(24)
    obs3 = torch.rand(24)
    
    global_obs = torch.cat([obs1, obs2, obs3]).unsqueeze(0)
    agent_id = torch.tensor([0])
    
    val1 = critic(global_obs, agent_id)
    act1 = actor(obs1)
    
    # Perturb agent 2's obs (obs2)
    obs2_perturbed = obs2 + 1.0
    global_obs_perturbed = torch.cat([obs1, obs2_perturbed, obs3]).unsqueeze(0)
    
    val2 = critic(global_obs_perturbed, agent_id)
    act2 = actor(obs1)
    
    assert torch.abs(val1 - val2) > 1e-5, "Critic output should change when other agent's obs changes"
    assert torch.allclose(act1, act2), "Actor output should NOT change"

def test_gae_hand_case():
    rewards = np.array([1.0, 1.0])
    values = np.array([0.0, 0.0, 0.0])
    dones = np.array([False, True])
    gamma = 0.99
    lam = 0.95
    
    adv = compute_gae(rewards, values, dones, gamma, lam)
    
    np.testing.assert_allclose(adv, [1.9405, 1.0], rtol=1e-5)

def test_masking_perturbation():
    # Agent finishes at step 2 of a 6-step rollout
    # 0-indexed: step 0 (active), step 1 (finishes, done=True), step 2+ (masked out)
    # So step 0 and 1 are active. Steps 2, 3, 4, 5 are masked.
    
    # Construct a fake batch and test if changing step 3 affects the loss
    actor = SharedActor(10, 3)
    critic = CentralizedCritic(10, 1)
    
    # Simulating the exact dataset preparation from train_mappo
    step = 6
    N = 1
    obs_dim = 10
    
    obs_buf = torch.zeros((step, N, obs_dim))
    actions_buf = torch.zeros((step, N, 3))
    logprobs_buf = torch.zeros((step, N))
    rewards_buf = torch.ones((step, N))
    dones_buf = torch.zeros((step, N))
    dones_buf[1, 0] = 1.0 # finishes at step index 1
    values_buf = torch.zeros((step, N))
    next_values = torch.zeros(N)
    
    global_obs_buf = obs_buf.view(step, N * obs_dim)
    
    def get_loss(rewards_arr, values_arr, actions_arr):
        active_mask = torch.ones((step, N), dtype=torch.bool)
        agent_dones = dones_buf[:, 0].numpy()
        done_idx = np.where(agent_dones)[0]
        first_done = done_idx[0]
        active_mask[first_done+1:, 0] = False
        
        adv = compute_gae(rewards_arr[:, 0].numpy(), 
                          torch.cat([values_arr[:, 0], next_values]).numpy(), 
                          agent_dones, 0.99, 0.95)
        
        advantages = torch.tensor(adv, dtype=torch.float32).unsqueeze(1)
        returns = advantages + values_arr
        
        flat_obs = obs_buf[active_mask]
        flat_actions = actions_arr[active_mask]
        flat_logprobs = logprobs_buf[active_mask]
        flat_advantages = advantages[active_mask]
        flat_returns = returns[active_mask]
        
        flat_agent_ids = torch.zeros(len(flat_obs), dtype=torch.long)
        flat_global_obs = global_obs_buf.unsqueeze(1).expand(step, N, N * obs_dim)[active_mask]
        
        dataset = {
            'obs': flat_obs,
            'global_obs': flat_global_obs,
            'agent_id': flat_agent_ids,
            'actions': flat_actions,
            'old_logprobs': flat_logprobs,
            'advantages': flat_advantages,
            'returns': flat_returns
        }
        
        return compute_losses(actor, critic, dataset, 0.2)
        
    p_loss1, v_loss1 = get_loss(rewards_buf, values_buf, actions_buf)
    
    # Perturb > 2 (e.g. step index 3)
    rewards_buf2 = rewards_buf.clone()
    rewards_buf2[3, 0] = 100.0
    values_buf2 = values_buf.clone()
    values_buf2[3, 0] = 100.0
    actions_buf2 = actions_buf.clone()
    actions_buf2[3, 0, 0] = 100.0
    
    p_loss2, v_loss2 = get_loss(rewards_buf2, values_buf2, actions_buf2)
    
    assert torch.abs(p_loss1 - p_loss2) < 1e-6
    assert torch.abs(v_loss1 - v_loss2) < 1e-6
    
    # Perturb <= 1 (e.g. step index 0)
    rewards_buf3 = rewards_buf.clone()
    rewards_buf3[0, 0] = 100.0
    p_loss3, v_loss3 = get_loss(rewards_buf3, values_buf, actions_buf)
    
    assert torch.abs(p_loss1 - p_loss3) > 1e-6
    

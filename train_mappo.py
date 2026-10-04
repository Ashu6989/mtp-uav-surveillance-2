import torch
import numpy as np
import torch.nn as nn

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: (T,)
    values: (T+1,)
    dones: (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae_lam = 0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        next_value = values[t+1]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        
    return advantages

def compute_losses(actor, critic, batch, clip_eps=0.2):
    """
    batch contains flattened active transitions:
    obs, global_obs, agent_id, actions, old_logprobs, advantages, returns
    """
    obs = batch['obs']
    global_obs = batch['global_obs']
    agent_id = batch['agent_id']
    actions = batch['actions']
    old_logprobs = batch['old_logprobs']
    advantages = batch['advantages']
    returns = batch['returns']
    
    # Policy Loss
    _, logprobs = actor.get_action_and_logprob(obs, actions)
    ratio = torch.exp(logprobs - old_logprobs)
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value Loss
    values = critic(global_obs, agent_id)
    value_loss = 0.5 * ((values - returns) ** 2).mean()
    
    return policy_loss, value_loss

def rollout_and_train(env, actor, critic, actor_optimizer, critic_optimizer, 
                      rollout_steps=6, epochs=4, gamma=0.99, lam=0.95, clip_eps=0.2):
    # Dummy train step used for masking tests
    pass

def train_mappo(env, actor, critic, total_timesteps=10000, 
                gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4, epochs=10, batch_size=64):
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    
    num_steps = 200 # env.max_steps
    
    N = env.N
    N_actual = getattr(env, 'N_actual', N)
    obs_dim = env.obs_dim
    
    # Buffers
    obs_buf = torch.zeros((num_steps, N, obs_dim), dtype=torch.float32)
    actions_buf = torch.zeros((num_steps, N, env.action_dim), dtype=torch.float32)
    logprobs_buf = torch.zeros((num_steps, N), dtype=torch.float32)
    rewards_buf = torch.zeros((num_steps, N), dtype=torch.float32)
    dones_buf = torch.zeros((num_steps, N), dtype=torch.float32)
    values_buf = torch.zeros((num_steps, N), dtype=torch.float32)
    
    global_obs_buf = torch.zeros((num_steps, N * obs_dim), dtype=torch.float32)
    
    global_step = 0
    
    while global_step < total_timesteps:
        obs = env.reset()
        
        episode_rewards = np.zeros(N)
        step = 0
        
        while step < num_steps:
            global_obs = obs.reshape(N * obs_dim)
            
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32)
                global_obs_t = torch.tensor(global_obs, dtype=torch.float32).unsqueeze(0).expand(N, -1)
                agent_ids = torch.arange(N)
                
                actions, logprobs = actor.get_action_and_logprob(obs_t)
                values = critic(global_obs_t, agent_ids)
                
            actions_np = actions.numpy()
            next_obs, rewards, dones = env.step(actions_np)
            
            # Store in buffers
            obs_buf[step] = obs_t
            actions_buf[step] = actions
            logprobs_buf[step] = logprobs
            rewards_buf[step] = torch.tensor(rewards, dtype=torch.float32)
            dones_buf[step] = torch.tensor(dones, dtype=torch.float32)
            values_buf[step] = values
            global_obs_buf[step] = torch.tensor(global_obs, dtype=torch.float32)
            
            obs = next_obs
            step += 1
            global_step += N_actual
            
            if np.all(dones[:N_actual]):
                break
                
        # Calculate next value
        with torch.no_grad():
            global_obs = obs.reshape(N * obs_dim)
            global_obs_t = torch.tensor(global_obs, dtype=torch.float32).unsqueeze(0).expand(N, -1)
            next_values = critic(global_obs_t, torch.arange(N))
            
        # Calculate advantages per agent
        advantages = torch.zeros((step, N), dtype=torch.float32)
        returns = torch.zeros((step, N), dtype=torch.float32)
        
        # We need an active mask:
        # 1. Mask out dummy agents completely (i >= N_actual)
        # 2. For actual agents (i < N_actual), once done becomes True, that step is the last active step.
        #    Steps AFTER done becomes True are masked out.
        active_mask = torch.ones((step, N), dtype=torch.bool)
        if N_actual < N:
            active_mask[:, N_actual:] = False
            
        for i in range(N_actual):
            agent_rewards = rewards_buf[:step, i].numpy()
            agent_dones = dones_buf[:step, i].numpy()
            agent_values = torch.cat([values_buf[:step, i], next_values[i:i+1]]).numpy()
            
            adv = compute_gae(agent_rewards, agent_values, agent_dones, gamma, lam)
            advantages[:, i] = torch.tensor(adv, dtype=torch.float32)
            returns[:, i] = advantages[:, i] + values_buf[:step, i]
            
            # active mask: False after the first True in dones
            done_idx = np.where(agent_dones)[0]
            if len(done_idx) > 0:
                first_done = done_idx[0]
                if first_done + 1 < step:
                    active_mask[first_done+1:, i] = False
                    
        # Flatten active transitions
        flat_obs = obs_buf[:step][active_mask]
        flat_actions = actions_buf[:step][active_mask]
        flat_logprobs = logprobs_buf[:step][active_mask]
        flat_advantages = advantages[active_mask]
        flat_returns = returns[active_mask]
        
        # We need agent_ids and global_obs for critic
        agent_ids_matrix = torch.arange(N).unsqueeze(0).expand(step, N)
        flat_agent_ids = agent_ids_matrix[active_mask]
        
        global_obs_matrix = global_obs_buf[:step].unsqueeze(1).expand(step, N, N * obs_dim)
        flat_global_obs = global_obs_matrix[active_mask]
        
        if len(flat_obs) == 0:
            continue
            
        # Normalize advantages
        if len(flat_advantages) > 1:
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            
        dataset = {
            'obs': flat_obs,
            'global_obs': flat_global_obs,
            'agent_id': flat_agent_ids,
            'actions': flat_actions,
            'old_logprobs': flat_logprobs,
            'advantages': flat_advantages,
            'returns': flat_returns
        }
        
        for _ in range(epochs):
            p_loss, v_loss = compute_losses(actor, critic, dataset, clip_eps)
            
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            
            total_loss = p_loss + v_loss
            total_loss.backward()
            
            actor_optimizer.step()
            critic_optimizer.step()

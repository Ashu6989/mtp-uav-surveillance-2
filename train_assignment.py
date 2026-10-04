import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from warehouse import Warehouse
from viewpoints import generate_roi_cells, generate_candidates, thin_viewpoints
from visibility import compute_visibility_matrix
from assignment_env import AssignmentEnv, mask_fn

def create_default_assignment_env(k_g=1):
    """
    Constructs the canonical 20x15x5 warehouse environment with 4 obstacle racks,
    ROI patches, thinned candidate viewpoints, visibility matrix, and 2 UAV starts.
    """
    dimensions = (20, 15, 5)
    resolution = 0.25
    obstacles = [
        ((2, 4, 0), (10, 5, 3)),
        ((12, 4, 0), (18, 5, 3)),
        ((2, 10, 0), (10, 11, 3)),
        ((12, 10, 0), (18, 11, 3))
    ]
    warehouse = Warehouse(dimensions, obstacles, resolution)
    
    patches = [
        {'x': (2, 10), 'y': (5, 5), 'z': (0, 3), 'normal': (0, 1, 0)},
        {'x': (12, 18), 'y': (5, 5), 'z': (0, 3), 'normal': (0, 1, 0)},
        {'x': (2, 10), 'y': (10, 10), 'z': (0, 3), 'normal': (0, -1, 0)},
        {'x': (12, 18), 'y': (10, 10), 'z': (0, 3), 'normal': (0, -1, 0)}
    ]
    roi_cells = generate_roi_cells(patches, spacing=0.5)
    
    valid_candidates, _ = generate_candidates(warehouse, roi_cells, [1.5, 3.0, 5.0], [-1, 0, 1], 0.5)
    viewpoints = thin_viewpoints(warehouse, valid_candidates, 1.0)
    
    a_mg = compute_visibility_matrix(viewpoints, roi_cells, warehouse)
    uav_starts = [(1, 7.5, 1), (19, 7.5, 1)]
    
    env = AssignmentEnv(uav_starts, viewpoints, roi_cells, a_mg, k_g, warehouse)
    return env

def evaluate_policy(model, masked_env, n_episodes=20):
    """
    Runs deterministic evaluation over n_episodes and returns
    mean total reward, mean K-coverage %, and success rate (C_K == 1.0).
    """
    total_rewards = []
    coverages = []
    successes = []
    
    for ep in range(n_episodes):
        obs, _ = masked_env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        
        while not done:
            action_masks = masked_env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, _, info = masked_env.step(int(action))
            ep_reward += reward
            
        c_k = info.get('C_K', 0.0)
        total_rewards.append(ep_reward)
        coverages.append(c_k)
        successes.append(1.0 if c_k >= 1.0 else 0.0)
        
    mean_reward = float(np.mean(total_rewards))
    mean_coverage_pct = float(np.mean(coverages) * 100.0)
    success_rate_pct = float(np.mean(successes) * 100.0)
    
    return {
        'mean_reward': mean_reward,
        'mean_coverage_pct': mean_coverage_pct,
        'success_rate_pct': success_rate_pct,
        'total_rewards': total_rewards,
        'coverages': coverages
    }

def train_assignment(total_timesteps=50000, learning_rate=3e-4, gamma=0.99,
                     save_path="models/ppo_assignment.zip", eval_episodes=20,
                     env=None, seed=42):
    """
    Trains MaskablePPO on AssignmentEnv wrapped with ActionMasker,
    saves the trained model, and evaluates performance.
    """
    if env is None:
        env = create_default_assignment_env()
        
    # 1. Wrap AssignmentEnv with ActionMasker
    masked_env = ActionMasker(env, mask_fn)
    
    # 2. Train MaskablePPO("MlpPolicy", ...)
    model = MaskablePPO(
        "MlpPolicy",
        masked_env,
        gamma=gamma,
        learning_rate=learning_rate,
        verbose=1,
        seed=seed
    )
    
    print(f"Training MaskablePPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    print("Training complete.")
    
    # 3. Save the trained model
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        model.save(save_path)
        print(f"Model successfully saved to {save_path}")
        
    # 4. Run an evaluation over test episodes
    print(f"\nRunning evaluation over {eval_episodes} test episodes...")
    metrics = evaluate_policy(model, masked_env, n_episodes=eval_episodes)
    
    print("================ Evaluation Results ================")
    print(f"Mean Total Reward:      {metrics['mean_reward']:.3f}")
    print(f"Mean K-Coverage:        {metrics['mean_coverage_pct']:.2f}%")
    print(f"Success Rate (C_K=1.0): {metrics['success_rate_pct']:.2f}%")
    print("====================================================")
    
    return model, metrics

if __name__ == "__main__":
    train_assignment(
        total_timesteps=50000,
        learning_rate=3e-4,
        gamma=0.99,
        save_path="models/ppo_assignment.zip",
        eval_episodes=20
    )

import gymnasium as gym
from gymnasium import spaces
import numpy as np

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()

class AssignmentEnv(gym.Env):
    def __init__(self, uav_starts, viewpoints, roi_cells, a_mg, k_g, warehouse, 
                 w_C=1.0, w_L=0.1, w_minus=1.0, B=50.0):
        super().__init__()
        
        self.uav_starts = np.array(uav_starts)
        self.N = len(self.uav_starts)
        self.P = len(viewpoints)
        self.G = len(roi_cells)
        
        self.a_mg = np.array(a_mg, dtype=bool)
        self.k_g = k_g
        self.w_C = w_C
        self.w_L = w_L
        self.w_minus = w_minus
        self.B = B
        
        self.warehouse = warehouse
        self.viewpoints = viewpoints
        
        self.obs_dim = self.G + self.P + 3 + self.P
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.action_space = spaces.Discrete(self.P)
        
        self.vp_pos = np.array([vp.position for vp in self.viewpoints]) if self.P > 0 else np.zeros((0,3))
        self.vp_regions = np.array([vp.region for vp in self.viewpoints]) if self.P > 0 else np.zeros(0)
        
        # Region of each UAV is derived from warehouse
        self.uav_regions = np.array([self.warehouse.region_of(s) for s in self.uav_starts])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_uav_idx = 0
        self.z_g = np.zeros(self.G, dtype=int)
        self.mu = np.zeros(self.P, dtype=int)
        
        self._advance_to_valid_uav()
        
        return self._get_obs(), {}
        
    def _advance_to_valid_uav(self):
        while self.current_uav_idx < self.N:
            if np.any(self.action_masks()):
                break
            self.current_uav_idx += 1
            
    def _get_obs(self):
        if self.current_uav_idx >= self.N:
            s_u = np.zeros(3)
            d_u = np.zeros(self.P)
        else:
            s_u = self.uav_starts[self.current_uav_idx]
            d_u = np.linalg.norm(self.vp_pos - s_u, axis=1)
            
        z_hat_g = np.minimum(self.z_g, self.k_g) / self.k_g
        
        obs = np.concatenate([z_hat_g, self.mu, s_u, d_u]).astype(np.float32)
        return obs
        
    def action_masks(self):
        if self.current_uav_idx >= self.N:
            return np.zeros(self.P, dtype=bool)
            
        uav_region = self.uav_regions[self.current_uav_idx]
        mask = (self.mu == 0) & (self.vp_regions == uav_region)
        return mask
        
    def step(self, action):
        if self.current_uav_idx >= self.N:
            return self._get_obs(), 0.0, True, False, {}
            
        # We allow taking invalid actions and just penalizing if standard PPO without masks somehow tests it,
        # but with MaskablePPO this shouldn't happen.
        if not self.action_masks()[action]:
            # Invalid action
            return self._get_obs(), -10.0, True, False, {}
            
        self.mu[action] = 1
        z_g_old = self.z_g.copy()
        self.z_g += self.a_mg[action].astype(int)
        
        s_u = self.uav_starts[self.current_uav_idx]
        d_u_m = np.linalg.norm(self.vp_pos[action] - s_u)
        
        deltaC_t = np.sum(np.minimum(self.z_g, self.k_g) - np.minimum(z_g_old, self.k_g))
        r_t = self.w_C * deltaC_t - self.w_L * d_u_m
        
        self.current_uav_idx += 1
        self._advance_to_valid_uav()
        
        terminated = (self.current_uav_idx >= self.N)
        
        info = {}
        if terminated:
            C_K = np.sum(self.z_g >= self.k_g) / self.G
            s_minus = np.maximum(0, self.k_g - self.z_g)
            S_minus = np.sum(s_minus)
            
            if C_K == 1.0:
                R_terminal = self.B
            else:
                R_terminal = -self.w_minus * S_minus
                
            r_t += R_terminal
            
            info['C_K'] = C_K
            info['S_minus'] = S_minus
            info['z_g'] = self.z_g.copy()
            
        return self._get_obs(), r_t, terminated, False, info

def train_assignment_policy(env, total_timesteps=20000):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    
    # Wrap environment with ActionMasker
    masked_env = ActionMasker(env, mask_fn)
    
    model = MaskablePPO("MlpPolicy", masked_env, gamma=0.99, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model

import numpy as np

class MAPPOEnv:
    def __init__(self, warehouse, reference_paths,
                 V_max=2.0, dt=0.5, d_safe=0.5, lambda1=10.0, lambda2=10.0, alpha=0.05,
                 waypoint_reach_threshold=0.3, max_steps=200, n_nearest=2, k_waypoints_ahead=3):
        self.warehouse = warehouse
        self.reference_paths = reference_paths
        self.N = len(reference_paths)
        
        self.V_max = V_max
        self.dt = dt
        self.d_safe = d_safe
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.waypoint_reach_threshold = waypoint_reach_threshold
        self.max_steps = max_steps
        self.n_nearest = n_nearest
        self.k_waypoints_ahead = k_waypoints_ahead
        
        self.obs_dim = 3 + 3 + 3 * self.k_waypoints_ahead + 3 * self.n_nearest + 3
        self.action_dim = 3
        
        self.reset()
        
    def reset(self):
        self.current_step = 0
        
        self.positions = np.array([path[0] for path in self.reference_paths], dtype=float)
        self.velocities = np.zeros((self.N, 3), dtype=float)
        
        # Start tracking the second waypoint (index 1) if it exists, else 0
        self.waypoint_indices = np.array([1 if len(path) > 1 else 0 for path in self.reference_paths], dtype=int)
        self.dones = np.zeros(self.N, dtype=bool)
        
        return self._get_obs()
        
    def _nearest_obstacle_point(self, p):
        if not self.warehouse.obstacles:
            return p
            
        min_dist = float('inf')
        nearest_pt = p
        
        for (min_c, max_c) in self.warehouse.obstacles:
            clamped = np.clip(p, min_c, max_c)
            dist = np.linalg.norm(clamped - p)
            if dist < min_dist:
                min_dist = dist
                nearest_pt = clamped
                
        return nearest_pt

    def _get_obs(self):
        obs = np.zeros((self.N, self.obs_dim), dtype=float)
        
        for i in range(self.N):
            p_i = self.positions[i]
            v_i = self.velocities[i]
            
            # Next k waypoints
            waypoints_ahead = []
            curr_wp_idx = self.waypoint_indices[i]
            path = self.reference_paths[i]
            
            for k in range(self.k_waypoints_ahead):
                idx = curr_wp_idx + k
                if idx < len(path):
                    waypoints_ahead.append(path[idx])
                else:
                    waypoints_ahead.append(path[-1]) # pad with final
                    
            wp_flat = np.array(waypoints_ahead).flatten()
            
            # Relative positions of n_nearest
            rel_positions = []
            if self.N > 1:
                dists = []
                for j in range(self.N):
                    if i != j:
                        dist = np.linalg.norm(self.positions[j] - p_i)
                        dists.append((dist, self.positions[j] - p_i))
                dists.sort(key=lambda x: x[0])
                for j in range(min(self.n_nearest, len(dists))):
                    rel_positions.extend(dists[j][1])
                    
            # Pad if fewer neighbors
            while len(rel_positions) < 3 * self.n_nearest:
                rel_positions.extend([1000.0, 1000.0, 1000.0])
                
            # Nearest obstacle point
            nearest_obs = self._nearest_obstacle_point(p_i)
            
            obs[i] = np.concatenate([p_i, v_i, wp_flat, rel_positions, nearest_obs])
            
        return obs
        
    def step(self, actions):
        actions = np.array(actions)
        rewards = np.zeros(self.N, dtype=float)
        
        new_velocities = self.velocities.copy()
        for i in range(self.N):
            if not self.dones[i]:
                v_new = self.velocities[i] + actions[i]
                norm = np.linalg.norm(v_new)
                if norm > self.V_max:
                    v_new = v_new / norm * self.V_max
                new_velocities[i] = v_new
            else:
                new_velocities[i] = np.zeros(3) # Frozen
                
        new_positions = self.positions.copy()
        for i in range(self.N):
            if not self.dones[i]:
                new_positions[i] = self.positions[i] + new_velocities[i] * self.dt
                
        for i in range(self.N):
            if self.dones[i]:
                continue
                
            p_post = new_positions[i]
            v_post = new_velocities[i]
            
            path = self.reference_paths[i]
            w_target_idx = min(self.waypoint_indices[i], len(path) - 1)
            w_target = np.array(path[w_target_idx])
            
            tracking_cost = np.linalg.norm(p_post - w_target)
            
            inter_uav_penalty = 0.0
            for j in range(self.N):
                if i != j:
                    if np.linalg.norm(p_post - new_positions[j]) < self.d_safe:
                        inter_uav_penalty = self.lambda1
                        break
                        
            obs_penalty = 0.0
            if self.warehouse.distance_to_obstacles(p_post) < self.d_safe:
                obs_penalty = self.lambda2
                
            energy_cost = self.alpha * np.linalg.norm(v_post) * self.dt
            
            rewards[i] = -tracking_cost - inter_uav_penalty - obs_penalty - energy_cost
            
        self.velocities = new_velocities
        self.positions = new_positions
        
        for i in range(self.N):
            if self.dones[i]:
                continue
                
            path = self.reference_paths[i]
            w_target_idx = min(self.waypoint_indices[i], len(path) - 1)
            w_target = np.array(path[w_target_idx])
            
            if np.linalg.norm(self.positions[i] - w_target) <= self.waypoint_reach_threshold:
                if self.waypoint_indices[i] < len(path) - 1:
                    self.waypoint_indices[i] += 1
                else:
                    self.dones[i] = True
                    
        self.current_step += 1
        if self.current_step >= self.max_steps:
            for i in range(self.N):
                if not self.dones[i]:
                    self.dones[i] = True
                    
        return self._get_obs(), rewards, self.dones.copy()

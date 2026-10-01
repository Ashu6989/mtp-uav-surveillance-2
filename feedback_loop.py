import numpy as np

class DistanceCache:
    def __init__(self, A_star_func, max_cost=1000.0):
        self.A_star_func = A_star_func
        self.max_cost = max_cost
        self.cache_mappo = {}
        self.cache_astar = {}
        
    def get_distance(self, uav_start, viewpoint):
        key = (tuple(np.round(uav_start, 4)), tuple(np.round(viewpoint, 4)))
        
        if key in self.cache_mappo:
            return self.cache_mappo[key]
            
        if key in self.cache_astar:
            return self.cache_astar[key]
            
        path, length = self.A_star_func(uav_start, viewpoint)
        if path is None:
            length = self.max_cost
            
        self.cache_astar[key] = length
        return length
        
    def update_mappo_distance(self, uav_start, viewpoint, actual_length):
        key = (tuple(np.round(uav_start, 4)), tuple(np.round(viewpoint, 4)))
        self.cache_mappo[key] = actual_length

class JointFeedbackLoop:
    def __init__(self, distance_cache, mappo_retrain_freq=500):
        self.distance_cache = distance_cache
        self.mappo_retrain_freq = mappo_retrain_freq
        self.ppo_episodes_since_retrain = 0
        
    def on_ppo_episode_end(self, flown_trajectories):
        """
        flown_trajectories: list of dicts {'start': pt, 'viewpoint': pt, 'actual_length': L}
        """
        self.ppo_episodes_since_retrain += 1
        for traj in flown_trajectories:
            self.distance_cache.update_mappo_distance(traj['start'], traj['viewpoint'], traj['actual_length'])
            
    def should_retrain_mappo(self):
        return self.ppo_episodes_since_retrain >= self.mappo_retrain_freq
        
    def reset_retrain_counter(self):
        self.ppo_episodes_since_retrain = 0

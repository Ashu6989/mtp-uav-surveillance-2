import numpy as np
from scipy.ndimage import label

class Warehouse:
    def __init__(self, dimensions, obstacles, resolution):
        self.dimensions = tuple(dimensions)
        self.resolution = resolution
        self.obstacles = obstacles
        
        self.grid_shape = (
            int(round(self.dimensions[0] / resolution)),
            int(round(self.dimensions[1] / resolution)),
            int(round(self.dimensions[2] / resolution))
        )
        
        self.occupancy_grid = np.zeros(self.grid_shape, dtype=bool)
        
        for obs in self.obstacles:
            min_c, max_c = obs
            
            min_idx = np.floor(np.array(min_c) / resolution).astype(int)
            max_idx = np.ceil(np.array(max_c) / resolution).astype(int)
            
            min_idx = np.clip(min_idx, 0, self.grid_shape)
            max_idx = np.clip(max_idx, 0, self.grid_shape)
            
            self.occupancy_grid[
                min_idx[0]:max_idx[0],
                min_idx[1]:max_idx[1],
                min_idx[2]:max_idx[2]
            ] = True

        structure = np.ones((3, 3, 3), dtype=int)
        labeled, num_features = label(~self.occupancy_grid, structure)
        self.regions = labeled - 1

    def _world_to_grid(self, point):
        idx = np.floor(np.array(point) / self.resolution).astype(int)
        idx = np.clip(idx, 0, np.array(self.grid_shape) - 1)
        return tuple(idx)

    def region_of(self, point):
        return self.regions[self._world_to_grid(point)]
        
    def is_free(self, point):
        x, y, z = point
        if not (0 <= x <= self.dimensions[0] and 
                0 <= y <= self.dimensions[1] and 
                0 <= z <= self.dimensions[2]):
            return False
            
        for (min_c, max_c) in self.obstacles:
            if (min_c[0] <= x <= max_c[0] and
                min_c[1] <= y <= max_c[1] and
                min_c[2] <= z <= max_c[2]):
                return False
        return True

    def distance_to_obstacles(self, point):
        point = np.array(point)
        min_dist = float('inf')
        
        if not self.obstacles:
            return float('inf')
            
        for (min_c, max_c) in self.obstacles:
            min_c = np.array(min_c)
            max_c = np.array(max_c)
            
            d = np.maximum(0, min_c - point) + np.maximum(0, point - max_c)
            dist = np.linalg.norm(d)
            if dist < min_dist:
                min_dist = dist
                
        return min_dist

    def is_valid_position(self, point, d_safe):
        if not self.is_free(point):
            return False
        return self.distance_to_obstacles(point) >= d_safe

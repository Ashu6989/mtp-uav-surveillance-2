import numpy as np
import heapq
import warnings

def is_segment_safe(p1, p2, warehouse, d_safe):
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-6:
        return warehouse.is_valid_position(p1, d_safe)
        
    num_samples = max(2, int(np.ceil(dist / 0.05))) # dense sampling
    samples = np.linspace(p1, p2, num_samples)
    for p in samples:
        if not warehouse.is_valid_position(p, d_safe):
            return False
    return True

def snap_to_valid_voxel(point, warehouse, d_safe, max_radius=10):
    start_idx = warehouse._world_to_grid(point)
    start_center = (np.array(start_idx) + 0.5) * warehouse.resolution
    
    if warehouse.is_valid_position(start_center, d_safe):
        return start_idx, start_center
        
    # BFS to find nearest valid voxel
    queue = [start_idx]
    visited = set([start_idx])
    
    while queue:
        curr = queue.pop(0)
        curr_center = (np.array(curr) + 0.5) * warehouse.resolution
        
        if warehouse.is_valid_position(curr_center, d_safe):
            return curr, curr_center
            
        # Add neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nxt = (curr[0]+dx, curr[1]+dy, curr[2]+dz)
                    
                    # check bounds
                    if (0 <= nxt[0] < warehouse.grid_shape[0] and
                        0 <= nxt[1] < warehouse.grid_shape[1] and
                        0 <= nxt[2] < warehouse.grid_shape[2]):
                        
                        if nxt not in visited:
                            # limit radius
                            if max(abs(nxt[0]-start_idx[0]), abs(nxt[1]-start_idx[1]), abs(nxt[2]-start_idx[2])) <= max_radius:
                                visited.add(nxt)
                                queue.append(nxt)
                                
    return None, None

def plan_reference_path(warehouse, start, goal, d_safe):
    start = np.array(start)
    goal = np.array(goal)
    
    # 1. Start/Goal Snapping
    start_unsafe = not warehouse.is_valid_position(start, d_safe)
    goal_unsafe = not warehouse.is_valid_position(goal, d_safe)
    
    if start_unsafe:
        warnings.warn(f"Start position {start} is unsafe. Snapping to nearest valid voxel.")
    if goal_unsafe:
        warnings.warn(f"Goal position {goal} is unsafe. Snapping to nearest valid voxel.")
        
    start_idx, start_center = snap_to_valid_voxel(start, warehouse, d_safe)
    goal_idx, goal_center = snap_to_valid_voxel(goal, warehouse, d_safe)
    
    if start_idx is None or goal_idx is None:
        return None, 0.0 # Cannot snap
        
    # Different regions?
    if warehouse.region_of(start_center) != warehouse.region_of(goal_center):
        return None, 0.0
        
    # 2. A* Search
    def heuristic(a, b):
        pa = (np.array(a) + 0.5) * warehouse.resolution
        pb = (np.array(b) + 0.5) * warehouse.resolution
        return np.linalg.norm(pb - pa)
        
    open_set = []
    heapq.heappush(open_set, (0.0, start_idx))
    
    came_from = {}
    g_score = {start_idx: 0.0}
    f_score = {start_idx: heuristic(start_idx, goal_idx)}
    
    valid_cache = {}
    
    def is_valid_idx(idx):
        if idx not in valid_cache:
            pt = (np.array(idx) + 0.5) * warehouse.resolution
            valid_cache[idx] = warehouse.is_valid_position(pt, d_safe)
        return valid_cache[idx]

    path_found = False
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_idx:
            path_found = True
            break
            
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                        
                    neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)
                    
                    if not (0 <= neighbor[0] < warehouse.grid_shape[0] and
                            0 <= neighbor[1] < warehouse.grid_shape[1] and
                            0 <= neighbor[2] < warehouse.grid_shape[2]):
                        continue
                        
                    if not is_valid_idx(neighbor):
                        continue
                        
                    # Euclidean cost between voxel centers
                    cost = np.linalg.norm(np.array([dx, dy, dz]) * warehouse.resolution)
                    tentative_g_score = g_score[current] + cost
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_idx)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
    if not path_found:
        return None, 0.0
        
    # Reconstruct path
    path_indices = [goal_idx]
    curr = goal_idx
    while curr in came_from:
        curr = came_from[curr]
        path_indices.append(curr)
    path_indices.reverse()
    
    raw_path = [(np.array(idx) + 0.5) * warehouse.resolution for idx in path_indices]
    
    # 3. Handle raw endpoints
    if not start_unsafe:
        raw_path.insert(0, start)
    if not goal_unsafe:
        raw_path.append(goal)
        
    # 4. Simplification (Greedy straight-line shortcutting)
    simplified_path = [raw_path[0]]
    curr_idx = 0
    n = len(raw_path)
    
    while curr_idx < n - 1:
        # Search backwards from end
        for j in range(n - 1, curr_idx, -1):
            if is_segment_safe(raw_path[curr_idx], raw_path[j], warehouse, d_safe):
                simplified_path.append(raw_path[j])
                curr_idx = j
                break
                
    # Calculate length
    path_length = 0.0
    for i in range(len(simplified_path) - 1):
        path_length += np.linalg.norm(simplified_path[i+1] - simplified_path[i])
        
    # Convert points to tuples/lists
    simplified_path = [tuple(p) for p in simplified_path]
    return simplified_path, path_length

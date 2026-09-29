import numpy as np

def is_los_free(p1, p2, obstacles, eps=1e-4):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    d = p2 - p1
    
    for min_c, max_c in obstacles:
        tmin = 0.0
        tmax = 1.0
        
        intersect = True
        for i in range(3):
            if abs(d[i]) < 1e-9:
                if p1[i] < min_c[i] or p1[i] > max_c[i]:
                    intersect = False
                    break
            else:
                inv_d = 1.0 / d[i]
                t1 = (min_c[i] - p1[i]) * inv_d
                t2 = (max_c[i] - p1[i]) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    intersect = False
                    break
                    
        # Exclude intersections strictly at the endpoints to prevent ROI cells 
        # from being blocked by their own surface.
        if intersect and tmin <= 1.0 - eps and tmax >= eps:
            return False
            
    return True

def compute_visibility_matrix(viewpoints, roi_cells, warehouse, r_max=8.0, theta_max_deg=75.0):
    """
    Computes the visibility matrix a_mg.
    Returns: np.ndarray bool of shape (P, G)
    """
    P = len(viewpoints)
    G = len(roi_cells)
    
    a_mg = np.zeros((P, G), dtype=bool)
    
    cos_theta_max = np.cos(np.radians(theta_max_deg))
    
    vp_pos = np.array([vp.position for vp in viewpoints])
    cell_pos = np.array([cell.position for cell in roi_cells])
    cell_norms = np.array([cell.normal for cell in roi_cells])
    
    obstacles = warehouse.obstacles
    
    for m in range(P):
        vm = vp_pos[m]
        
        diffs = vm - cell_pos
        dists = np.linalg.norm(diffs, axis=1)
        
        # Range check
        range_mask = dists <= r_max
        
        # Incidence angle check
        valid_dists_mask = dists > 1e-6
        dirs = np.zeros_like(diffs)
        dirs[valid_dists_mask] = diffs[valid_dists_mask] / dists[valid_dists_mask, np.newaxis]
        
        dots = np.sum(cell_norms * dirs, axis=1)
        incidence_mask = dots >= cos_theta_max
        
        # Combine masks
        candidate_g = np.where(range_mask & incidence_mask)[0]
        
        # LOS check (only for valid range + incidence)
        for g in candidate_g:
            rg = cell_pos[g]
            if is_los_free(vm, rg, obstacles):
                a_mg[m, g] = True
                
    return a_mg

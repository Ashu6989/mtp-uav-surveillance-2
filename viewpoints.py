import numpy as np

class ROICell:
    def __init__(self, position, normal):
        self.position = np.array(position, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(self.normal)

class Viewpoint:
    def __init__(self, position, region):
        self.position = np.array(position, dtype=float)
        self.region = region

def generate_roi_cells(patches, spacing=0.5):
    """
    patches is a list of dicts: {'x': (min, max), 'y': (min, max), 'z': (min, max), 'normal': (nx, ny, nz)}
    """
    cells = []
    for patch in patches:
        x_min, x_max = patch['x']
        y_min, y_max = patch['y']
        z_min, z_max = patch['z']
        normal = patch['normal']
        
        if x_min == x_max:
            xs = [x_min]
        else:
            xs = np.arange(x_min + spacing/2, x_max, spacing)
            
        if y_min == y_max:
            ys = [y_min]
        else:
            ys = np.arange(y_min + spacing/2, y_max, spacing)
            
        if z_min == z_max:
            zs = [z_min]
        else:
            zs = np.arange(z_min + spacing/2, z_max, spacing)
            
        for x in xs:
            for y in ys:
                for z in zs:
                    cells.append(ROICell((x, y, z), normal))
    return cells

def generate_candidates(warehouse, roi_cells, standoffs, lateral_offsets, d_safe):
    valid_candidates = []
    points_tested = 0
    
    for cell in roi_cells:
        n = cell.normal
        # Deterministic tangent vectors for consistent ordering
        if abs(n[1]) > 0.9: # Normal along Y
            t1 = np.array([1.0, 0.0, 0.0])
        elif abs(n[0]) > 0.9: # Normal along X
            t1 = np.array([0.0, 1.0, 0.0])
        else: # Normal along Z or others
            t1 = np.array([1.0, 0.0, 0.0])
            
        t1 = t1 - np.dot(t1, n) * n
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        
        for standoff in standoffs:
            for off1 in lateral_offsets:
                for off2 in lateral_offsets:
                    points_tested += 1
                    p = cell.position + standoff * n + off1 * t1 + off2 * t2
                    if warehouse.is_valid_position(p, d_safe):
                        valid_candidates.append(p)
                        
    return valid_candidates, points_tested

def thin_viewpoints(warehouse, valid_candidates, min_spacing):
    thinned = []
    
    if len(valid_candidates) == 0:
        return []
        
    for p in valid_candidates:
        if not thinned:
            thinned.append(p)
        else:
            diffs = np.array(thinned) - p
            dists = np.linalg.norm(diffs, axis=1)
            if np.all(dists >= min_spacing):
                thinned.append(p)
                
    viewpoints = []
    for p in thinned:
        region = warehouse.region_of(p)
        viewpoints.append(Viewpoint(p, region))
        
    return viewpoints

import numpy as np

def compute_coverage_metrics(z_g, k_g):
    G = len(z_g)
    C_K = np.sum(z_g >= k_g) / G
    
    s_minus = np.maximum(0, k_g - z_g)
    S_minus = np.sum(s_minus)
    
    s_plus = np.maximum(0, z_g - k_g)
    S_plus = np.sum(s_plus)
    
    C_1 = np.sum(z_g >= 1) / G
    B = 1.0 - C_1
    
    return C_K, S_minus, S_plus, C_1, B

def compute_trajectory_metrics(trajectories, alpha=0.05, V_max=2.0):
    """
    trajectories: list of list of points
    """
    L_tot = 0.0
    T_mission = 0.0
    E_tot = 0.0
    
    for path in trajectories:
        L_u = 0.0
        for i in range(len(path) - 1):
            L_u += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            
        E_u = alpha * L_u
        T_u = L_u / V_max 
        
        L_tot += L_u
        E_tot += E_u
        T_mission = max(T_mission, T_u)
        
    return L_tot, T_mission, E_tot

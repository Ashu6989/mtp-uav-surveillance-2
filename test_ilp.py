import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ilp_baseline import solve_assignment_ilp
from viewpoints import Viewpoint

def test_toy_scenario_ilp():
    uav_starts = [(0,0,0), (10,0,0)]
    
    viewpoints = [
        Viewpoint((2,2,0), 0),
        Viewpoint((5,5,0), 0),
        Viewpoint((8,2,0), 0)
    ]
    
    a_mg = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ], dtype=bool)
    
    k_g = 2
    
    distance_matrix = np.array([
        [2.828, 7.071, 8.246],
        [8.246, 7.071, 2.828]
    ])
    
    res_hard = solve_assignment_ilp(uav_starts, viewpoints, a_mg, k_g, distance_matrix, hard_coverage=True)
    assert res_hard is None, "Hard coverage k_g=2 with 2 UAVs should be infeasible"
    
    res_soft = solve_assignment_ilp(uav_starts, viewpoints, a_mg, k_g, distance_matrix, w_minus=1.0, hard_coverage=False)
    assert res_soft is not None
    assert abs(res_soft['S_minus'] - 3.0) < 1e-5, f"Expected S_minus=3, got {res_soft['S_minus']}"

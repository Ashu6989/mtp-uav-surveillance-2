import pytest
import numpy as np
from evaluation import compute_coverage_metrics, compute_trajectory_metrics

def test_toy_scenario_actions():
    z_g = np.array([1,2,1,1])
    C_K, S_minus, S_plus, C_1, B = compute_coverage_metrics(z_g, k_g=2)
    assert C_K == 0.25
    assert S_minus == 3
    assert S_plus == 0
    assert C_1 == 1.0
    assert B == 0.0
    
    z_g = np.array([1,1,1,1])
    C_K, S_minus, S_plus, C_1, B = compute_coverage_metrics(z_g, k_g=2)
    assert C_K == 0.0
    assert S_minus == 4
    assert S_plus == 0
    assert C_1 == 1.0
    assert B == 0.0

def test_trajectory_metrics():
    trajectories = [[(0,0,0), (10,0,0)]]
    L_tot, T_mission, E_tot = compute_trajectory_metrics(trajectories, alpha=0.05, V_max=2.0)
    assert L_tot == 10.0
    assert E_tot == 0.5
    assert T_mission == 5.0

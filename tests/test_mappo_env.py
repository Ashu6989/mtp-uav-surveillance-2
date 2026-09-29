import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warehouse import Warehouse
from mappo_env import MAPPOEnv

@pytest.fixture
def test_warehouse():
    dimensions = (20, 15, 5)
    resolution = 0.25
    obstacles = [
        ((2, 4, 0), (10, 5, 3)),
        ((12, 4, 0), (18, 5, 3)),
        ((2, 10, 0), (10, 11, 3)),
        ((12, 10, 0), (18, 11, 3))
    ]
    return Warehouse(dimensions, obstacles, resolution)

def test_single_uav_standard(test_warehouse):
    reference_paths = [[(1, 7.5, 1), (3, 7.5, 1)]]
    env = MAPPOEnv(test_warehouse, reference_paths)
    
    # Check observation dimension
    assert env.obs_dim == 24, f"Expected obs_dim 24, got {env.obs_dim}"
    
    obs = env.reset()
    
    actions = [[0.5, 0.0, 0.0]]
    obs, rewards, dones = env.step(actions)
    
    np.testing.assert_allclose(env.velocities[0], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(env.positions[0], [1.25, 7.5, 1.0])
    np.testing.assert_allclose(rewards[0], -1.7625)

def test_single_uav_clipping(test_warehouse):
    reference_paths = [[(1, 7.5, 1), (3, 7.5, 1)]]
    env = MAPPOEnv(test_warehouse, reference_paths)
    env.reset()
    
    actions = [[10.0, 0.0, 0.0]]
    obs, rewards, dones = env.step(actions)
    
    np.testing.assert_allclose(env.velocities[0], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(env.positions[0], [2.0, 7.5, 1.0])
    np.testing.assert_allclose(rewards[0], -1.05)

def test_single_uav_obstacle_penalty(test_warehouse):
    reference_paths = [[(5, 3.6, 1), (5, 2.6, 1)]]
    env = MAPPOEnv(test_warehouse, reference_paths)
    env.reset()
    
    actions = [[0.0, 0.0, 0.0]]
    obs, rewards, dones = env.step(actions)
    
    np.testing.assert_allclose(rewards[0], -11.0)

def test_two_uavs_inter_penalty(test_warehouse):
    reference_paths = [
        [(5, 7.5, 1), (9, 7.5, 1)],
        [(5.3, 7.5, 1), (9, 7.5, 1)]
    ]
    env = MAPPOEnv(test_warehouse, reference_paths)
    env.reset()
    
    actions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    obs, rewards, dones = env.step(actions)
    
    np.testing.assert_allclose(rewards[0], -14.0)
    np.testing.assert_allclose(rewards[1], -13.7)

def test_done_agent_frozen(test_warehouse):
    reference_paths = [
        [(1, 7.5, 1), (1.1, 7.5, 1)], # Ends quickly
        [(5, 7.5, 1), (9, 7.5, 1)]
    ]
    env = MAPPOEnv(test_warehouse, reference_paths, waypoint_reach_threshold=0.3)
    env.reset()
    
    # Step 1: UAV0 finishes
    actions = [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]
    obs, rewards, dones = env.step(actions)
    
    assert dones[0] == True
    assert dones[1] == False
    
    # Step 2: Test frozen
    actions = [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    obs, rewards, dones = env.step(actions)
    
    np.testing.assert_allclose(env.velocities[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(env.positions[0], [1.1, 7.5, 1.0])
    assert rewards[0] == 0.0
    
    # Also verify it stays in neighbor slots
    # UAV 1's observation should include UAV 0's relative position
    obs_uav1 = obs[1]
    rel_pos_start = 3 + 3 + 3 * 3
    rel_pos_uav0 = obs_uav1[rel_pos_start : rel_pos_start + 3]
    
    expected_rel_pos = np.array([1.1, 7.5, 1.0]) - env.positions[1]
    np.testing.assert_allclose(rel_pos_uav0, expected_rel_pos)

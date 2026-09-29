import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warehouse import Warehouse
from viewpoints import generate_roi_cells, generate_candidates, thin_viewpoints, ROICell, Viewpoint
from visibility import compute_visibility_matrix
from assignment_env import AssignmentEnv, mask_fn, train_assignment_policy
from sb3_contrib.common.wrappers import ActionMasker
import random

class MockWarehouse:
    def region_of(self, point):
        return 0

@pytest.fixture
def toy_env():
    uav_starts = [(0,0,0), (10,0,0)]
    viewpoints = [
        Viewpoint((2,2,0), 0),
        Viewpoint((5,5,0), 0),
        Viewpoint((8,2,0), 0)
    ]
    roi_cells = [ROICell((0,0,0), (0,1,0))] * 4 # Mock 4 cells
    
    a_mg = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ], dtype=bool)
    
    k_g = 2
    w_C = 1.0
    w_L = 0.1
    w_minus = 1.0
    B = 50.0
    
    warehouse = MockWarehouse()
    
    env = AssignmentEnv(uav_starts, viewpoints, roi_cells, a_mg, k_g, warehouse,
                        w_C=w_C, w_L=w_L, w_minus=w_minus, B=B)
    return env

def test_toy_scenario_actions_0_2(toy_env):
    obs, _ = toy_env.reset()
    
    # Action 0
    obs, r1, done, _, info = toy_env.step(0)
    assert abs(r1 - 1.717) < 1e-3, f"Expected 1.717, got {r1}"
    assert not done
    
    # Action 2
    obs, r2, done, _, info = toy_env.step(2)
    # Step reward = 1.717, terminal = -4 -> total -2.283
    # Actually, r2 should be the step reward + terminal reward = 1.71716 - 4 = -2.2828
    assert abs(r2 - (-2.283)) < 1e-3, f"Expected -2.283, got {r2}"
    assert done
    
    episode_return = r1 + r2
    assert abs(episode_return - (-0.566)) < 1e-3
    assert info['C_K'] == 0.0
    assert info['S_minus'] == 4

def test_toy_scenario_actions_0_1(toy_env):
    obs, _ = toy_env.reset()
    
    # Action 0
    obs, r1, done, _, info = toy_env.step(0)
    
    # Action 1
    obs, r2, done, _, info = toy_env.step(1)
    # r2 step reward = 2.293, terminal = -3
    assert abs(r2 - (-0.707)) < 1e-3
    assert done
    
    episode_return = r1 + r2
    assert abs(episode_return - 1.010) < 1e-3
    assert info['C_K'] == 0.25
    assert info['S_minus'] == 3

def test_mask_and_skip():
    # Test mask: after viewpoint 0 is taken, action 0 is invalid for next UAV
    uav_starts = [(0,0,0), (10,0,0)]
    viewpoints = [Viewpoint((2,2,0), 0), Viewpoint((5,5,0), 0)]
    roi_cells = [ROICell((0,0,0), (0,1,0))]
    a_mg = np.array([[1], [1]])
    
    env = AssignmentEnv(uav_starts, viewpoints, roi_cells, a_mg, 1, MockWarehouse())
    env.reset()
    
    masks = env.action_masks()
    assert masks[0] == True
    assert masks[1] == True
    
    env.step(0)
    
    masks = env.action_masks()
    assert masks[0] == False
    assert masks[1] == True
    
    # Test skip: A UAV with no same-region viewpoint is skipped without crashing
    uav_starts = [(0,0,0), (10,0,0)]
    viewpoints = [Viewpoint((2,2,0), 1)] # Region 1
    
    class MultiRegionWarehouse:
        def region_of(self, point):
            if point[0] < 5:
                return 0
            return 1
            
    env = AssignmentEnv(uav_starts, viewpoints, roi_cells, np.array([[1]]), 1, MultiRegionWarehouse())
    
    # UAV 0 starts at (0,0,0) -> region 0. No viewpoints in region 0. It should be skipped automatically.
    # Therefore, after reset, current_uav_idx should be 1.
    obs, _ = env.reset()
    assert env.current_uav_idx == 1
    
def test_full_training_target():
    # Target: Fixture, UAV starts (1,7.5,1) and (19,7.5,1), k_g=1, maskable PPO ~20k steps
    
    dimensions = (20, 15, 5)
    resolution = 0.25
    obstacles = [
        ((2, 4, 0), (10, 5, 3)),
        ((12, 4, 0), (18, 5, 3)),
        ((2, 10, 0), (10, 11, 3)),
        ((12, 10, 0), (18, 11, 3))
    ]
    warehouse = Warehouse(dimensions, obstacles, resolution)
    
    patches = [
        {'x': (2, 10), 'y': (5, 5), 'z': (0, 3), 'normal': (0, 1, 0)},
        {'x': (12, 18), 'y': (5, 5), 'z': (0, 3), 'normal': (0, 1, 0)},
        {'x': (2, 10), 'y': (10, 10), 'z': (0, 3), 'normal': (0, -1, 0)},
        {'x': (12, 18), 'y': (10, 10), 'z': (0, 3), 'normal': (0, -1, 0)}
    ]
    roi_cells = generate_roi_cells(patches, spacing=0.5)
    
    valid_candidates, _ = generate_candidates(warehouse, roi_cells, [1.5, 3.0, 5.0], [-1, 0, 1], 0.5)
    viewpoints = thin_viewpoints(warehouse, valid_candidates, 1.0)
    
    a_mg = compute_visibility_matrix(viewpoints, roi_cells, warehouse)
    
    uav_starts = [(1, 7.5, 1), (19, 7.5, 1)]
    k_g = 1
    
    env = AssignmentEnv(uav_starts, viewpoints, roi_cells, a_mg, k_g, warehouse)
    
    # 1. Random masked episodes baseline
    masked_env = ActionMasker(env, mask_fn)
    
    random_c_k = []
    for _ in range(100):
        obs, _ = masked_env.reset()
        done = False
        while not done:
            masks = masked_env.action_masks()
            valid_actions = np.where(masks)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = 0 # Invalid, will be caught/skipped
            obs, reward, done, _, info = masked_env.step(action)
        random_c_k.append(info['C_K'])
        
    mean_random_c_k = np.mean(random_c_k)
    print(f"Mean Random C_K over 100 episodes: {mean_random_c_k}")
    
    # 2. Train maskable PPO
    model = train_assignment_policy(env, total_timesteps=20000)
    
    # 3. Evaluate deterministic trained policy
    obs, _ = masked_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True, action_masks=masked_env.action_masks())
        obs, reward, done, _, info = masked_env.step(int(action))
        
    trained_c_k = info['C_K']
    print(f"Trained deterministic policy C_K: {trained_c_k}")
    
    assert trained_c_k == 1.0, f"Expected trained C_K to be 1.0, got {trained_c_k}"
    assert trained_c_k > mean_random_c_k, "Trained policy did not beat random baseline!"

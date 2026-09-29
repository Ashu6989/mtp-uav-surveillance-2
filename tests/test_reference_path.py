import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warehouse import Warehouse
from reference_path import plan_reference_path

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

@pytest.fixture
def wall_warehouse():
    dimensions = (20, 15, 5)
    resolution = 0.25
    obstacles = [
        ((2, 4, 0), (10, 5, 3)),
        ((12, 4, 0), (18, 5, 3)),
        ((2, 10, 0), (10, 11, 3)),
        ((12, 10, 0), (18, 11, 3)),
        ((10, 0, 0), (10.5, 15, 5))
    ]
    return Warehouse(dimensions, obstacles, resolution)

def test_unobstructed_paths(test_warehouse):
    d_safe = 0.5
    
    # Path 1
    start1 = (1, 7.5, 1)
    goal1 = (11.2, 8.0, 0.2)
    straight_len1 = np.linalg.norm(np.array(goal1) - np.array(start1))
    
    path1, length1 = plan_reference_path(test_warehouse, start1, goal1, d_safe)
    
    assert abs(length1 - straight_len1) <= 0.05, f"Expected length {straight_len1}, got {length1}"
    assert len(path1) == 2, f"Expected 2 waypoints for straight path, got {len(path1)}"
    
    # Path 2
    start2 = (19, 7.5, 1)
    goal2 = (9.2, 8.0, 1.2)
    straight_len2 = np.linalg.norm(np.array(goal2) - np.array(start2))
    
    path2, length2 = plan_reference_path(test_warehouse, start2, goal2, d_safe)
    
    assert abs(length2 - straight_len2) <= 0.05, f"Expected length {straight_len2}, got {length2}"
    assert len(path2) == 2, f"Expected 2 waypoints for straight path, got {len(path2)}"

def test_obstructed_path(test_warehouse):
    d_safe = 0.5
    start = (5, 3, 1)
    goal = (5, 12, 1)
    
    path, length = plan_reference_path(test_warehouse, start, goal, d_safe)
    
    assert 11.5 <= length <= 14.5, f"Expected length in [11.5, 14.5], got {length}"
    assert 3 <= len(path) <= 10, f"Expected 3 to 10 waypoints, got {len(path)}"
    
    # Verify safety of simplified path
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        dist = np.linalg.norm(p2 - p1)
        num_samples = int(np.ceil(dist / 0.1))
        samples = np.linspace(p1, p2, num_samples + 1)
        
        for p in samples:
            d_obs = test_warehouse.distance_to_obstacles(p)
            assert d_obs >= d_safe - 1e-6, f"Distance {d_obs} < {d_safe} at point {p}"

def test_snapping(test_warehouse):
    d_safe = 0.5
    # 0.2 m from R1 (unsafe)
    start_unsafe = (5, 3.8, 1)
    goal = (5, 1, 1) # safe goal
    
    with pytest.warns(UserWarning, match="is unsafe"):
        path, length = plan_reference_path(test_warehouse, start_unsafe, goal, d_safe)
    
    assert path is not None
    assert np.linalg.norm(np.array(path[0]) - np.array(start_unsafe)) > 1e-3, "Path should not begin at unsafe raw point"
    
    # Check all waypoints are safe
    for p in path:
        assert test_warehouse.is_valid_position(p, d_safe)

def test_different_regions(wall_warehouse):
    d_safe = 0.5
    start = (5, 7.5, 1) # Region 1
    goal = (15, 7.5, 1) # Region 2
    
    path, length = plan_reference_path(wall_warehouse, start, goal, d_safe)
    
    assert path is None
    assert length == 0.0

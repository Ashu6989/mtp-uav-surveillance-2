import pytest
from warehouse import Warehouse

def get_fixture_warehouse():
    dimensions = (20, 15, 5)
    resolution = 0.25
    obstacles = [
        ((2, 4, 0), (10, 5, 3)),
        ((12, 4, 0), (18, 5, 3)),
        ((2, 10, 0), (10, 11, 3)),
        ((12, 10, 0), (18, 11, 3))
    ]
    return Warehouse(dimensions, obstacles, resolution)

def test_single_region():
    w = get_fixture_warehouse()
    assert w.regions.max() == 0, f"Expected 1 region, got {w.regions.max() + 1}"

def test_wall_two_regions():
    dimensions = (20, 15, 5)
    resolution = 0.25
    obstacles = [
        ((2, 4, 0), (10, 5, 3)),
        ((12, 4, 0), (18, 5, 3)),
        ((2, 10, 0), (10, 11, 3)),
        ((12, 10, 0), (18, 11, 3)),
        ((10, 0, 0), (10.5, 15, 5))
    ]
    w = Warehouse(dimensions, obstacles, resolution)
    
    assert w.regions.max() == 1, f"Expected 2 regions, got {w.regions.max() + 1}"
    
    r1 = w.region_of((5, 7.5, 1))
    r2 = w.region_of((15, 7.5, 1))
    
    assert r1 >= 0
    assert r2 >= 0
    assert r1 != r2

def test_boundary_clamp():
    w = get_fixture_warehouse()
    r1 = w.region_of((20, 7.5, 1))
    r2 = w.region_of((20, 15, 5))
    
    assert r1 >= 0, f"Expected r1 >= 0, got {r1}"
    assert r2 >= 0, f"Expected r2 >= 0, got {r2}"

def test_distance_and_validity():
    w = get_fixture_warehouse()
    
    dist = w.distance_to_obstacles((5, 7.5, 1))
    assert abs(dist - 2.5) < 1e-6, f"Expected 2.5, got {dist}"
    
    assert not w.is_valid_position((5, 3.7, 1), 0.5)
    assert w.is_valid_position((5, 3.7, 1), 0.2)
    
    assert not w.is_free((5, 4.5, 1))

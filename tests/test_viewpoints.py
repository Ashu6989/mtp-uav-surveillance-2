import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warehouse import Warehouse
from viewpoints import generate_roi_cells, generate_candidates, thin_viewpoints

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
def test_patches():
    return [
        {'x': (2, 10), 'y': (5, 5), 'z': (0, 3), 'normal': (0, 1, 0)},
        {'x': (12, 18), 'y': (5, 5), 'z': (0, 3), 'normal': (0, 1, 0)},
        {'x': (2, 10), 'y': (10, 10), 'z': (0, 3), 'normal': (0, -1, 0)},
        {'x': (12, 18), 'y': (10, 10), 'z': (0, 3), 'normal': (0, -1, 0)}
    ]

def test_roi_cells(test_patches):
    cells = generate_roi_cells(test_patches, spacing=0.5)
    
    assert len(cells) == 336, f"Expected 336 ROI cells, got {len(cells)}"
    
    # Check normals
    for cell in cells:
        y_pos = cell.position[1]
        if y_pos == 5.0:
            np.testing.assert_array_equal(cell.normal, [0, 1, 0])
        elif y_pos == 10.0:
            np.testing.assert_array_equal(cell.normal, [0, -1, 0])
        else:
            pytest.fail(f"Unexpected cell y position: {y_pos}")

def test_candidate_generation_and_thinning(test_warehouse, test_patches):
    cells = generate_roi_cells(test_patches, spacing=0.5)
    
    d_safe = 0.5
    standoffs = [1.5, 3.0, 5.0]
    lateral_offsets = [-1, 0, 1]
    
    valid_candidates, points_tested = generate_candidates(
        test_warehouse, cells, standoffs, lateral_offsets, d_safe
    )
    
    assert points_tested == 9072, f"Expected 9072 points tested, got {points_tested}"
    
    # Check that all surviving candidates are valid
    for cand in valid_candidates:
        assert test_warehouse.is_valid_position(cand, d_safe)
        
    # Check count is ~5664 (within 10%)
    assert 5664 * 0.9 <= len(valid_candidates) <= 5664 * 1.1, f"Expected ~5664 valid candidates, got {len(valid_candidates)}"
    print(f"Valid candidates count: {len(valid_candidates)}")
    
    # Thinning
    min_spacing = 1.0
    viewpoints = thin_viewpoints(test_warehouse, valid_candidates, min_spacing)
    
    # Check count is ~197 (within 10%)
    assert 197 * 0.9 <= len(viewpoints) <= 197 * 1.1, f"Expected ~197 viewpoints after thinning, got {len(viewpoints)}"
    print(f"Thinned viewpoints count: {len(viewpoints)}")
    
    # Check region 0
    for vp in viewpoints:
        assert vp.region == 0, f"Expected region 0, got {vp.region}"

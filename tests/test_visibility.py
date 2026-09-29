import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warehouse import Warehouse
from viewpoints import generate_roi_cells, generate_candidates, thin_viewpoints, ROICell, Viewpoint
from visibility import compute_visibility_matrix, is_los_free

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
def test_cells():
    patches = [
        {'x': (2, 10), 'y': (5, 5), 'z': (0, 3), 'normal': (0, 1, 0)},
        {'x': (12, 18), 'y': (5, 5), 'z': (0, 3), 'normal': (0, 1, 0)},
        {'x': (2, 10), 'y': (10, 10), 'z': (0, 3), 'normal': (0, -1, 0)},
        {'x': (12, 18), 'y': (10, 10), 'z': (0, 3), 'normal': (0, -1, 0)}
    ]
    return generate_roi_cells(patches, spacing=0.5)

def test_visibility_specific_cell(test_warehouse):
    cell = ROICell((6.25, 5.0, 1.75), (0, 1, 0))
    vp1 = Viewpoint((6.25, 7.5, 1.75), 0)
    vp2 = Viewpoint((6.25, 12.0, 1.75), 0)
    vp3 = Viewpoint((6.25, 3.0, 1.75), 0)
    
    matrix = compute_visibility_matrix([vp1, vp2, vp3], [cell], test_warehouse, r_max=8.0, theta_max_deg=75.0)
    
    assert matrix[0, 0] == True, "Should be visible from (6.25, 7.5, 1.75)"
    assert matrix[1, 0] == False, "Should NOT be visible from (6.25, 12.0, 1.75) (LOS blocked by R3)"
    assert matrix[2, 0] == False, "Should NOT be visible from (6.25, 3.0, 1.75) (behind own rack)"

def test_full_visibility_matrix(test_warehouse, test_cells):
    d_safe = 0.5
    standoffs = [1.5, 3.0, 5.0]
    lateral_offsets = [-1, 0, 1]
    
    valid_candidates, _ = generate_candidates(
        test_warehouse, test_cells, standoffs, lateral_offsets, d_safe
    )
    
    viewpoints = thin_viewpoints(test_warehouse, valid_candidates, min_spacing=1.0)
    
    matrix = compute_visibility_matrix(viewpoints, test_cells, test_warehouse, r_max=8.0, theta_max_deg=75.0)
    
    P, G = matrix.shape
    assert P == len(viewpoints)
    assert G == len(test_cells)
    
    visible_pairs = np.sum(matrix)
    print(f"Visible pairs: {visible_pairs}")
    
    visible_per_cell = np.sum(matrix, axis=0)
    zero_visibility_cells = np.sum(visible_per_cell == 0)
    print(f"Zero visibility cells: {zero_visibility_cells}")
    assert zero_visibility_cells == 0, f"Expected 0 ROI cells with zero visibility, got {zero_visibility_cells}"
    
    cells_per_vp = np.mean(np.sum(matrix, axis=1))
    vps_per_cell = np.mean(visible_per_cell)
    
    print(f"Cells per viewpoint: {cells_per_vp:.2f}")
    print(f"Viewpoints per cell: {vps_per_cell:.2f}")
    
    assert 34600 * 0.85 <= visible_pairs <= 34600 * 1.15
    assert 176 * 0.85 <= cells_per_vp <= 176 * 1.15
    assert 103 * 0.85 <= vps_per_cell <= 103 * 1.15

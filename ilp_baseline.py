from ortools.linear_solver import pywraplp
import numpy as np

def solve_assignment_ilp(uav_starts, viewpoints, a_mg, k_g, distance_matrix,
                         w_L=0.1, w_minus=1.0, w_plus=0.0, 
                         hard_coverage=False):
    """
    ILP baseline for exact viewpoint assignment optimization.
    """
    N = len(uav_starts)
    P = len(viewpoints)
    G = a_mg.shape[1]
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None
        
    # Variables
    x = {}
    for u in range(N):
        for m in range(P):
            x[u, m] = solver.IntVar(0, 1, f'x_{u}_{m}')
            
    # Each UAV selects exactly 1
    for u in range(N):
        solver.Add(solver.Sum([x[u, m] for m in range(P)]) == 1)
        
    # Distinct viewpoints
    for m in range(P):
        solver.Add(solver.Sum([x[u, m] for u in range(N)]) <= 1)
        
    # Region eligibility constraint
    for u in range(N):
        for m in range(P):
            if distance_matrix[u, m] > 900.0:
                solver.Add(x[u, m] == 0)
                
    # Coverage
    s_minus = {}
    s_plus = {}
    for g in range(G):
        z_g_expr = solver.Sum([int(a_mg[m, g]) * x[u, m] for u in range(N) for m in range(P)])
        
        if hard_coverage:
            solver.Add(z_g_expr >= k_g)
        else:
            s_minus[g] = solver.NumVar(0, solver.infinity(), f's_minus_{g}')
            s_plus[g] = solver.NumVar(0, solver.infinity(), f's_plus_{g}')
            solver.Add(z_g_expr + s_minus[g] - s_plus[g] == k_g)
            
    # Objective
    objective = solver.Objective()
    objective.SetMinimization()
    
    for u in range(N):
        for m in range(P):
            objective.SetCoefficient(x[u, m], w_L * float(distance_matrix[u, m]))
            
    if not hard_coverage:
        for g in range(G):
            objective.SetCoefficient(s_minus[g], float(w_minus))
            objective.SetCoefficient(s_plus[g], float(w_plus))
            
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        assignment = []
        for u in range(N):
            for m in range(P):
                if x[u, m].solution_value() > 0.5:
                    assignment.append(m)
                    
        total_dist = sum(distance_matrix[u, assignment[u]] for u in range(N))
        
        S_minus_val = 0
        if not hard_coverage:
            S_minus_val = sum(s_minus[g].solution_value() for g in range(G))
            
        return {
            'assignment': assignment,
            'total_distance': total_dist,
            'S_minus': S_minus_val,
            'objective_value': objective.Value()
        }
    else:
        return None

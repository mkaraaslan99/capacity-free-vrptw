"""
Local Search Operators for Capacity-Free VRPTW
Implementation of Algorithms 3 and 4 from research paper
"""

import time
from typing import List, Tuple, Dict
from problems.vrptw import VRPTWInstance


def two_opt_local_search(vrptw_instance: VRPTWInstance, 
                       solution: List[List[int]],
                       max_iterations: int = 100,
                       max_time_seconds: float = 2.0,
                       max_moves_evaluated: int = 20000) -> Tuple[List[List[int]], float, Dict]:
    """
    2-opt local search operator (Algorithm 3)
    Improves individual routes by removing two edges and reconnecting
    
    Args:
        vrptw_instance: VRPTW problem instance
        solution: initial solution
        
    Returns:
        tuple of (improved_solution, total_cost, stats)
    """
    start_time = time.time()
    improved = True
    iterations = 0
    moves_evaluated = 0
    
    while improved and iterations < max_iterations:
        if (time.time() - start_time) >= max_time_seconds:
            break

        improved = False
        iterations += 1
        
        for route_idx, route in enumerate(solution):
            if (time.time() - start_time) >= max_time_seconds:
                break
            if moves_evaluated >= max_moves_evaluated:
                break
            if len(route) < 2:  # Need at least 2 customers for 2-opt
                continue

            old_cost = vrptw_instance.calculate_route_distance(route)
                
            # Try all 2-opt moves
            for i in range(len(route) - 1):
                if (time.time() - start_time) >= max_time_seconds:
                    break
                for j in range(i + 1, len(route)):
                    moves_evaluated += 1
                    if moves_evaluated >= max_moves_evaluated:
                        break
                    if (time.time() - start_time) >= max_time_seconds:
                        break
                    # Apply 2-opt move
                    new_route = apply_two_opt(route, i, j)
                    
                    # Check feasibility
                    feasible, _ = vrptw_instance.is_route_feasible(new_route)
                    
                    if feasible:
                        # Calculate cost improvement
                        new_cost = vrptw_instance.calculate_route_distance(new_route)
                        
                        if new_cost < old_cost:
                            # Accept improvement
                            solution[route_idx] = new_route
                            improved = True
                            break
                
                if improved:
                    break
            
            if improved:
                break
    
    total_cost = vrptw_instance.calculate_solution_cost(solution)
    
    stats = {
        'algorithm': '2-opt Local Search',
        'time': time.time() - start_time,
        'iterations': iterations,
        'moves_evaluated': moves_evaluated,
        'routes': len(solution),
        'total_cost': total_cost,
        'feasible': vrptw_instance.is_solution_feasible(solution)[0]
    }
    
    return solution, total_cost, stats


def apply_two_opt(route: List[int], i: int, j: int) -> List[int]:
    """
    Apply 2-opt move to route by reversing segment between i and j
    
    Args:
        route: original route
        i: first index (0-based)
        j: second index (0-based, j > i)
        
    Returns:
        new route after 2-opt
    """
    # Create new route: route[0:i] + reverse(route[i:j+1]) + route[j+1:]
    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
    return new_route


def relocation_local_search(vrptw_instance: VRPTWInstance,
                          solution: List[List[int]],
                          max_iterations: int = 50,
                          max_time_seconds: float = 3.0,
                          max_moves_evaluated: int = 50000) -> Tuple[List[List[int]], float, Dict]:
    """
    Relocation local search operator (Algorithm 4)
    Moves a single customer from one route to another
    
    Args:
        vrptw_instance: VRPTW problem instance
        solution: initial solution
        
    Returns:
        tuple of (improved_solution, total_cost, stats)
    """
    start_time = time.time()
    improved = True
    iterations = 0
    moves_evaluated = 0
    old_cost = vrptw_instance.calculate_solution_cost(solution)
    
    while improved and iterations < max_iterations:
        if (time.time() - start_time) >= max_time_seconds:
            break
        improved = False
        iterations += 1
        
        # Try relocating each customer
        for route_a_idx, route_a in enumerate(solution):
            if (time.time() - start_time) >= max_time_seconds:
                break
            if moves_evaluated >= max_moves_evaluated:
                break
            for customer_idx, customer in enumerate(route_a):
                best_improvement = 0
                best_move = None
                
                # Try moving customer to all other routes and positions
                for route_b_idx, route_b in enumerate(solution):
                    if (time.time() - start_time) >= max_time_seconds:
                        break
                    if moves_evaluated >= max_moves_evaluated:
                        break
                    if route_a_idx == route_b_idx:
                        # Same route - try different positions
                        for pos in range(len(route_b)):
                            if pos == customer_idx:
                                continue
                            
                            new_solution = apply_relocation(vrptw_instance, solution, route_a_idx, route_b_idx, 
                                                       customer_idx, pos)
                            if new_solution:
                                moves_evaluated += 1
                                if moves_evaluated >= max_moves_evaluated:
                                    break
                                if (time.time() - start_time) >= max_time_seconds:
                                    break
                                new_cost = vrptw_instance.calculate_solution_cost(new_solution)
                                improvement = old_cost - new_cost
                                
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_move = (new_solution, route_a_idx, route_b_idx, 
                                               customer_idx, pos)
                    else:
                        # Different route - try all positions
                        for pos in range(len(route_b) + 1):  # +1 for insertion at end
                            new_solution = apply_relocation(vrptw_instance, solution, route_a_idx, route_b_idx,
                                                       customer_idx, pos)
                            if new_solution:
                                moves_evaluated += 1
                                if moves_evaluated >= max_moves_evaluated:
                                    break
                                if (time.time() - start_time) >= max_time_seconds:
                                    break
                                new_cost = vrptw_instance.calculate_solution_cost(new_solution)
                                improvement = old_cost - new_cost
                                
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_move = (new_solution, route_a_idx, route_b_idx,
                                               customer_idx, pos)
                
                # Apply best improvement if found
                if best_move and best_improvement > 0:
                    solution = best_move[0]
                    old_cost = old_cost - best_improvement
                    improved = True
                    break
            
            if improved:
                break
    
    total_cost = vrptw_instance.calculate_solution_cost(solution)
    
    stats = {
        'algorithm': 'Relocation Local Search',
        'time': time.time() - start_time,
        'iterations': iterations,
        'moves_evaluated': moves_evaluated,
        'routes': len(solution),
        'total_cost': total_cost,
        'feasible': vrptw_instance.is_solution_feasible(solution)[0]
    }
    
    return solution, total_cost, stats


def apply_relocation(vrptw_instance: VRPTWInstance,
                   solution: List[List[int]], 
                   route_a_idx: int, route_b_idx: int,
                   customer_idx: int, new_position: int) -> List[List[int]]:
    """
    Apply relocation move: move customer from route_a to route_b at new_position
    
    Args:
        solution: current solution
        route_a_idx: index of source route
        route_b_idx: index of target route
        customer_idx: index of customer to move in route_a
        new_position: position to insert in route_b
        
    Returns:
        new solution if feasible, None otherwise
    """
    # Create copy of solution
    new_solution = [route.copy() for route in solution]
    
    # Remove customer from route_a
    customer = new_solution[route_a_idx].pop(customer_idx)
    
    # Insert customer into route_b at new_position
    if route_a_idx == route_b_idx:
        # Same route - need to adjust position if customer_idx < new_position
        if customer_idx < new_position:
            new_position -= 1
    
    new_solution[route_b_idx].insert(new_position, customer)
    
    # Remove empty routes
    new_solution = [route for route in new_solution if route]

    # Reject if duplicates exist (or if customer is missing)
    served: List[int] = []
    for r in new_solution:
        served.extend(r)
    if len(served) != len(set(served)):
        return None

    # Reject if any route becomes time-window infeasible
    for r in new_solution:
        feasible, _ = vrptw_instance.is_route_feasible(r)
        if not feasible:
            return None
    
    return new_solution


def combined_local_search(vrptw_instance: VRPTWInstance,
                        solution: List[List[int]],
                        two_opt_max_iterations: int = 100,
                        two_opt_max_time_seconds: float = 2.0,
                        two_opt_max_moves_evaluated: int = 20000,
                        relocation_max_iterations: int = 50,
                        relocation_max_time_seconds: float = 3.0,
                        relocation_max_moves_evaluated: int = 50000) -> Tuple[List[List[int]], float, Dict]:
    """
    Combined local search: 2-opt followed by relocation
    Implements the improvement sequence from the dual-pipeline framework
    
    Args:
        vrptw_instance: VRPTW problem instance
        solution: initial solution
        
    Returns:
        tuple of (improved_solution, total_cost, stats)
    """
    start_time = time.time()
    initial_cost = vrptw_instance.calculate_solution_cost(solution)
    
    # Phase 1: 2-opt improvement
    solution_2opt, cost_2opt, stats_2opt = two_opt_local_search(
        vrptw_instance,
        solution,
        max_iterations=two_opt_max_iterations,
        max_time_seconds=two_opt_max_time_seconds,
        max_moves_evaluated=two_opt_max_moves_evaluated
    )
    
    # Phase 2: Relocation improvement
    final_solution, final_cost, stats_relocation = relocation_local_search(
        vrptw_instance,
        solution_2opt,
        max_iterations=relocation_max_iterations,
        max_time_seconds=relocation_max_time_seconds,
        max_moves_evaluated=relocation_max_moves_evaluated
    )
    
    # Combined statistics
    total_improvement = initial_cost - final_cost
    two_opt_improvement = initial_cost - cost_2opt
    relocation_improvement = cost_2opt - final_cost
    
    stats = {
        'algorithm': 'Combined Local Search',
        'time': time.time() - start_time,
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'total_improvement': total_improvement,
        'two_opt_improvement': two_opt_improvement,
        'relocation_improvement': relocation_improvement,
        'two_opt_iterations': stats_2opt.get('iterations'),
        'two_opt_moves_evaluated': stats_2opt.get('moves_evaluated'),
        'relocation_iterations': stats_relocation.get('iterations'),
        'relocation_moves_evaluated': stats_relocation.get('moves_evaluated'),
        'routes': len(final_solution),
        'feasible': vrptw_instance.is_solution_feasible(final_solution)[0]
    }
    
    return final_solution, final_cost, stats

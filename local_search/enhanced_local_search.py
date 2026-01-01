"""
Enhanced Local Search Operators for Capacity-Free VRPTW
Includes feasibility preservation and route splitting for better solutions
"""

import time
from typing import List, Tuple, Dict
from problems.vrptw import VRPTWInstance


def enhanced_two_opt_local_search(vrptw_instance: VRPTWInstance, 
                               solution: List[List[int]],
                               max_route_duration: float = 480) -> Tuple[List[List[int]], float, Dict]:
    """
    Enhanced 2-opt local search with feasibility preservation
    
    Args:
        vrptw_instance: VRPTW problem instance
        solution: initial solution
        max_route_duration: maximum allowed route duration (default 8 hours)
        
    Returns:
        tuple of (improved_solution, total_cost, stats)
    """
    start_time = time.time()
    improved = True
    iterations = 0
    
    while improved:
        improved = False
        iterations += 1
        
        for route_idx, route in enumerate(solution):
            if len(route) < 2:
                continue
                
            # Try all 2-opt moves
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Apply 2-opt move
                    new_route = apply_two_opt(route, i, j)
                    
                    # Check feasibility AND route duration
                    feasible, _ = vrptw_instance.is_route_feasible(new_route)
                    route_duration = calculate_route_duration(vrptw_instance, new_route)
                    
                    if feasible and route_duration <= max_route_duration:
                        # Calculate cost improvement
                        old_cost = vrptw_instance.calculate_route_distance(route)
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
        'algorithm': 'Enhanced 2-opt Local Search',
        'time': time.time() - start_time,
        'iterations': iterations,
        'routes': len(solution),
        'total_cost': total_cost,
        'feasible': vrptw_instance.is_solution_feasible(solution)[0]
    }
    
    return solution, total_cost, stats


def enhanced_relocation_local_search(vrptw_instance: VRPTWInstance,
                                solution: List[List[int]],
                                max_route_duration: float = 480) -> Tuple[List[List[int]], float, Dict]:
    """
    Enhanced relocation local search with feasibility preservation
    
    Args:
        vrptw_instance: VRPTW problem instance
        solution: initial solution
        max_route_duration: maximum allowed route duration
        
    Returns:
        tuple of (improved_solution, total_cost, stats)
    """
    start_time = time.time()
    improved = True
    iterations = 0
    
    while improved:
        improved = False
        iterations += 1
        
        # Try relocating each customer
        for route_a_idx, route_a in enumerate(solution):
            for customer_idx, customer in enumerate(route_a):
                best_improvement = 0
                best_move = None
                
                # Try moving customer to all other routes and positions
                for route_b_idx, route_b in enumerate(solution):
                    if route_a_idx == route_b_idx:
                        # Same route - try different positions
                        for pos in range(len(route_b)):
                            if pos == customer_idx:
                                continue
                            
                            new_solution = apply_relocation(solution, route_a_idx, route_b_idx,
                                                       customer_idx, pos)
                            if new_solution:
                                old_cost = vrptw_instance.calculate_solution_cost(solution)
                                new_cost = vrptw_instance.calculate_solution_cost(new_solution)
                                
                                # Check feasibility of all routes
                                feasible = vrptw_instance.is_solution_feasible(new_solution)[0]
                                all_routes_valid = all(
                                    calculate_route_duration(vrptw_instance, route) <= max_route_duration
                                    for route in new_solution
                                )
                                
                                if feasible and all_routes_valid:
                                    improvement = old_cost - new_cost
                                    
                                    if improvement > best_improvement:
                                        best_improvement = improvement
                                        best_move = (new_solution, route_a_idx, route_b_idx,
                                                   customer_idx, pos)
                    else:
                        # Different route - try all positions
                        for pos in range(len(route_b) + 1):
                            new_solution = apply_relocation(solution, route_a_idx, route_b_idx,
                                                       customer_idx, pos)
                            if new_solution:
                                old_cost = vrptw_instance.calculate_solution_cost(solution)
                                new_cost = vrptw_instance.calculate_solution_cost(new_solution)
                                
                                feasible = vrptw_instance.is_solution_feasible(new_solution)[0]
                                all_routes_valid = all(
                                    calculate_route_duration(vrptw_instance, route) <= max_route_duration
                                    for route in new_solution
                                )
                                
                                if feasible and all_routes_valid:
                                    improvement = old_cost - new_cost
                                    
                                    if improvement > best_improvement:
                                        best_improvement = improvement
                                        best_move = (new_solution, route_a_idx, route_b_idx,
                                                   customer_idx, pos)
                
                # Apply best improvement if found
                if best_move and best_improvement > 0:
                    solution = best_move[0]
                    improved = True
                    break
            
            if improved:
                break
    
    total_cost = vrptw_instance.calculate_solution_cost(solution)
    
    stats = {
        'algorithm': 'Enhanced Relocation Local Search',
        'time': time.time() - start_time,
        'iterations': iterations,
        'routes': len(solution),
        'total_cost': total_cost,
        'feasible': vrptw_instance.is_solution_feasible(solution)[0]
    }
    
    return solution, total_cost, stats


def route_splitting_local_search(vrptw_instance: VRPTWInstance,
                              solution: List[List[int]],
                              max_route_duration: float = 480) -> Tuple[List[List[int]], float, Dict]:
    """
    Route splitting local search to reduce route duration violations
    
    Args:
        vrptw_instance: VRPTW problem instance
        solution: initial solution
        max_route_duration: maximum allowed route duration
        
    Returns:
        tuple of (improved_solution, total_cost, stats)
    """
    start_time = time.time()
    improved = True
    iterations = 0
    
    while improved:
        improved = False
        iterations += 1
        
        # Check each route for duration violations
        for route_idx, route in enumerate(solution):
            route_duration = calculate_route_duration(vrptw_instance, route)
            
            if route_duration > max_route_duration:
                # Find best split point
                best_split = None
                best_cost = float('inf')
                
                for split_pos in range(1, len(route)):  # Don't split at first customer
                    # Split route into two
                    route1 = route[:split_pos]
                    route2 = route[split_pos:]
                    
                    # Check feasibility of both routes
                    feasible1, _ = vrptw_instance.is_route_feasible(route1)
                    feasible2, _ = vrptw_instance.is_route_feasible(route2)
                    
                    if feasible1 and feasible2:
                        # Calculate new total cost
                        old_cost = vrptw_instance.calculate_route_distance(route)
                        new_cost = (vrptw_instance.calculate_route_distance(route1) +
                                   vrptw_instance.calculate_route_distance(route2))
                        
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_split = (route1, route2)
                
                # Apply best split
                if best_split:
                    # Replace original route with split routes
                    solution.pop(route_idx)
                    solution.extend(best_split)
                    improved = True
                    break
        
        if improved:
            # Remove empty routes
            solution = [route for route in solution if route]
    
    total_cost = vrptw_instance.calculate_solution_cost(solution)
    
    stats = {
        'algorithm': 'Route Splitting Local Search',
        'time': time.time() - start_time,
        'iterations': iterations,
        'routes': len(solution),
        'total_cost': total_cost,
        'feasible': vrptw_instance.is_solution_feasible(solution)[0],
        'splits_performed': iterations
    }
    
    return solution, total_cost, stats


def enhanced_combined_local_search(vrptw_instance: VRPTWInstance,
                              solution: List[List[int]],
                              max_route_duration: float = 480) -> Tuple[List[List[int]], float, Dict]:
    """
    Enhanced combined local search with route splitting
    
    Args:
        vrptw_instance: VRPTW problem instance
        solution: initial solution
        max_route_duration: maximum allowed route duration
        
    Returns:
        tuple of (improved_solution, total_cost, stats)
    """
    start_time = time.time()
    initial_cost = vrptw_instance.calculate_solution_cost(solution)
    
    # Phase 1: Enhanced 2-opt with feasibility preservation
    solution_2opt, cost_2opt, stats_2opt = enhanced_two_opt_local_search(
        vrptw_instance, solution, max_route_duration)
    
    # Phase 2: Enhanced relocation with feasibility preservation
    solution_reloc, cost_reloc, stats_reloc = enhanced_relocation_local_search(
        vrptw_instance, solution_2opt, max_route_duration)
    
    # Phase 3: Route splitting for any remaining violations
    final_solution, final_cost, stats_split = route_splitting_local_search(
        vrptw_instance, solution_reloc, max_route_duration)
    
    # Combined statistics
    total_improvement = initial_cost - final_cost
    two_opt_improvement = initial_cost - cost_2opt
    relocation_improvement = cost_2opt - cost_reloc
    splitting_improvement = cost_reloc - final_cost
    
    stats = {
        'algorithm': 'Enhanced Combined Local Search',
        'time': time.time() - start_time,
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'total_improvement': total_improvement,
        'two_opt_improvement': two_opt_improvement,
        'relocation_improvement': relocation_improvement,
        'splitting_improvement': splitting_improvement,
        'routes': len(final_solution),
        'feasible': vrptw_instance.is_solution_feasible(final_solution)[0],
        'max_route_duration': max_route_duration
    }
    
    return final_solution, final_cost, stats


def calculate_route_duration(vrptw_instance: VRPTWInstance, route: List[int]) -> float:
    """
    Calculate total route duration including travel and service time
    
    Args:
        vrptw_instance: VRPTW problem instance
        route: list of customer IDs
        
    Returns:
        total route duration in minutes
    """
    if not route:
        return 0.0
    
    current_time = vrptw_instance.depot.ready_time
    current_node = 0  # Start at depot
    
    for customer_id in route:
        # Travel to customer
        travel_time = vrptw_instance.get_distance(current_node, customer_id)
        arrival_time = current_time + travel_time
        
        # Service start time (wait if early)
        service_start = max(arrival_time, vrptw_instance.nodes[customer_id].ready_time)
        
        # Update current time after service
        current_time = service_start + vrptw_instance.nodes[customer_id].service_time
        current_node = customer_id
    
    # Return to depot
    travel_time = vrptw_instance.get_distance(current_node, 0)
    return_time = current_time + travel_time
    
    # Total duration from depot ready time to return time
    return return_time - vrptw_instance.depot.ready_time


# Import helper functions from original module
def apply_two_opt(route: List[int], i: int, j: int) -> List[int]:
    """Apply 2-opt move to route by reversing segment between i and j"""
    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
    return new_route


def apply_relocation(solution: List[List[int]], 
                route_a_idx: int, route_b_idx: int,
                customer_idx: int, new_position: int) -> List[List[int]]:
    """Apply relocation move: move customer from route_a to route_b at new_position"""
    new_solution = [route.copy() for route in solution]
    
    # Remove customer from route_a
    customer = new_solution[route_a_idx].pop(customer_idx)
    
    # Insert customer into route_b at new_position
    if route_a_idx == route_b_idx:
        # Same route - adjust position if customer_idx < new_position
        if customer_idx < new_position:
            new_position -= 1
    
    new_solution[route_b_idx].insert(new_position, customer)
    
    # Remove empty routes
    new_solution = [route for route in new_solution if route]
    
    return new_solution

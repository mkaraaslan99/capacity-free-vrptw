"""
Clarke-Wright Savings Algorithm for Capacity-Free VRPTW
Implementation of Algorithm 2 from research paper
"""

import time
import random
from typing import List, Tuple, Dict, Optional
from problems.vrptw import VRPTWInstance


def clarke_wright_savings_vrptw(vrptw_instance: VRPTWInstance,
                                 max_vehicles: int = None,
                                 rcl_size: int = 1,
                                 seed: Optional[int] = None,
                                 rng: Optional[random.Random] = None) -> Tuple[List[List[int]], float, Dict]:
    """
    Clarke-Wright Savings heuristic for capacity-free VRPTW (Algorithm 2)
    
    Args:
        vrptw_instance: VRPTW problem instance
        max_vehicles: maximum number of vehicles (optional)
        
    Returns:
        tuple of (solution, total_cost, stats)
    """
    start_time = time.time()

    if rcl_size < 1:
        rcl_size = 1
    if rng is None:
        rng = random.Random(seed) if seed is not None else random.Random()
    
    # Phase 1: Initialization - create individual customer routes
    routes = []
    for customer_id in range(1, vrptw_instance.n_nodes):
        # Check if single customer route is feasible
        single_route = [customer_id]
        feasible, _ = vrptw_instance.is_route_feasible(single_route)
        if feasible:
            routes.append(single_route)
        else:
            # If not feasible, this customer cannot be served at all
            print(f"Warning: Customer {customer_id} cannot be served individually")
    
    # Phase 2: Savings computation
    savings = []
    for i in range(1, vrptw_instance.n_nodes):
        for j in range(i + 1, vrptw_instance.n_nodes):
            # Calculate savings: s = d0i + dj0 - dij
            s = (vrptw_instance.get_distance(0, i) + 
                 vrptw_instance.get_distance(j, 0) - 
                 vrptw_instance.get_distance(i, j))
            savings.append((s, i, j))
    
    # Sort savings in descending order
    savings.sort(reverse=True, key=lambda x: x[0])
    
    # Phase 3: Route merging
    merged_count = 0
    while savings:
        if rcl_size == 1:
            s, i, j = savings.pop(0)
        else:
            k = min(rcl_size, len(savings))
            pick_idx = rng.randrange(k)
            s, i, j = savings.pop(pick_idx)

        # Find routes containing i and j
        route_i = None
        route_j = None
        route_i_idx = None
        route_j_idx = None
        
        for idx, route in enumerate(routes):
            if i in route:
                route_i = route
                route_i_idx = idx
            if j in route:
                route_j = route
                route_j_idx = idx
        
        # Skip if i and j are in same route
        if route_i_idx == route_j_idx:
            continue
        
        # Check if routes can be merged (i at end of route_i, j at start of route_j)
        if route_i and route_j and route_i[-1] == i and route_j[0] == j:
            # Create merged route and check feasibility
            merged_route = route_i + route_j
            feasible, _ = vrptw_instance.is_route_feasible(merged_route)
            
            if feasible:
                # Perform merge
                routes[route_i_idx] = merged_route
                routes.pop(route_j_idx)
                merged_count += 1
                
                # Check vehicle limit
                if max_vehicles and len(routes) <= max_vehicles:
                    break
    
    # Calculate total cost
    total_cost = vrptw_instance.calculate_solution_cost(routes)
    
    # Statistics
    stats = {
        'algorithm': 'Clarke-Wright Savings VRPTW',
        'time': time.time() - start_time,
        'routes': len(routes),
        'customers_served': sum(len(route) for route in routes),
        'merged_count': merged_count,
        'feasible': vrptw_instance.is_solution_feasible(routes)[0],
        'rcl_size': rcl_size,
        'seed': seed
    }
    
    return routes, total_cost, stats


def parallel_savings_vrptw(vrptw_instance: VRPTWInstance,
                           max_vehicles: int = None) -> Tuple[List[List[int]], float, Dict]:
    """
    Parallel Savings heuristic for capacity-free VRPTW
    Enhanced version that considers all merge possibilities
    
    Args:
        vrptw_instance: VRPTW problem instance
        max_vehicles: maximum number of vehicles (optional)
        
    Returns:
        tuple of (solution, total_cost, stats)
    """
    start_time = time.time()
    
    # Phase 1: Initialization
    routes = []
    for customer_id in range(1, vrptw_instance.n_nodes):
        single_route = [customer_id]
        feasible, _ = vrptw_instance.is_route_feasible(single_route)
        if feasible:
            routes.append(single_route)
    
    # Phase 2: Enhanced savings computation
    savings = []
    for i in range(1, vrptw_instance.n_nodes):
        for j in range(i + 1, vrptw_instance.n_nodes):
            # Calculate savings for both merge directions
            s_ij = (vrptw_instance.get_distance(0, i) + 
                    vrptw_instance.get_distance(j, 0) - 
                    vrptw_instance.get_distance(i, j))
            s_ji = (vrptw_instance.get_distance(0, j) + 
                    vrptw_instance.get_distance(i, 0) - 
                    vrptw_instance.get_distance(j, i))
            
            savings.append((s_ij, i, j, 'i_to_j'))
            savings.append((s_ji, j, i, 'j_to_i'))
    
    # Sort savings in descending order
    savings.sort(reverse=True, key=lambda x: x[0])
    
    # Phase 3: Parallel route merging
    merged_count = 0
    for s, i, j, direction in savings:
        # Find routes containing i and j
        route_i = None
        route_j = None
        route_i_idx = None
        route_j_idx = None
        
        for idx, route in enumerate(routes):
            if i in route:
                route_i = route
                route_i_idx = idx
            if j in route:
                route_j = route
                route_j_idx = idx
        
        # Skip if i and j are in same route
        if route_i_idx == route_j_idx:
            continue
        
        # Check merge feasibility based on direction
        if direction == 'i_to_j':
            if can_merge_routes(vrptw_instance, route_i, route_j, i, j):
                merged_route = merge_routes(route_i, route_j, i, j)
                routes[route_i_idx] = merged_route
                routes.pop(route_j_idx)
                merged_count += 1
        else:  # j_to_i
            if can_merge_routes(vrptw_instance, route_j, route_i, j, i):
                merged_route = merge_routes(route_j, route_i, j, i)
                routes[route_j_idx] = merged_route
                routes.pop(route_i_idx)
                merged_count += 1
        
        # Check vehicle limit
        if max_vehicles and len(routes) <= max_vehicles:
            break
    
    # Calculate total cost
    total_cost = vrptw_instance.calculate_solution_cost(routes)
    
    # Statistics
    stats = {
        'algorithm': 'Parallel Savings VRPTW',
        'time': time.time() - start_time,
        'routes': len(routes),
        'customers_served': sum(len(route) for route in routes),
        'merged_count': merged_count,
        'feasible': vrptw_instance.is_solution_feasible(routes)[0]
    }
    
    return routes, total_cost, stats


def can_merge_routes(vrptw_instance: VRPTWInstance, 
                    route1: List[int], route2: List[int],
                    i: int, j: int) -> bool:
    """
    Check if two routes can be merged by connecting i to j
    
    Args:
        vrptw_instance: VRPTW problem instance
        route1: first route containing customer i
        route2: second route containing customer j
        i: customer at end of first route
        j: customer at beginning of second route
        
    Returns:
        True if merge is feasible
    """
    # Check if i is at the end of route1 and j is at the beginning of route2
    if route1[-1] != i or route2[0] != j:
        return False
    
    # Create merged route
    merged_route = route1 + route2
    
    # Check feasibility
    feasible, _ = vrptw_instance.is_route_feasible(merged_route)
    return feasible


def merge_routes(route1: List[int], route2: List[int], i: int, j: int) -> List[int]:
    """
    Merge two routes by connecting i to j
    
    Args:
        route1: first route ending with i
        route2: second route starting with j
        i: customer at end of first route
        j: customer at beginning of second route
        
    Returns:
        merged route
    """
    return route1 + route2


def find_route_containing_customer(routes: List[List[int]], customer: int) -> Tuple[List[int], int]:
    """
    Find route containing a specific customer
    
    Args:
        routes: list of routes
        customer: customer ID to find
        
    Returns:
        tuple of (route, route_index)
    """
    for idx, route in enumerate(routes):
        if customer in route:
            return route, idx
    return None, -1

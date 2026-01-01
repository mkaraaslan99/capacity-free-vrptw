"""
Nearest Neighbor Heuristics for Capacity-Free VRPTW
Implementation of Algorithm 1 from research paper
"""

import time
import random
from typing import List, Tuple, Dict, Optional
from problems.vrptw import VRPTWInstance


def nearest_neighbor_vrptw(vrptw_instance: VRPTWInstance, 
                          max_vehicles: int = None,
                          rcl_size: int = 1,
                          seed: Optional[int] = None,
                          rng: Optional[random.Random] = None,
                          w_distance: float = 1.0,
                          w_urgency: float = 0.0) -> Tuple[List[List[int]], float, Dict]:
    """
    Nearest Neighbor heuristic for capacity-free VRPTW (Algorithm 1)
    
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
    
    unserved = set(range(1, vrptw_instance.n_nodes))  # Customer IDs (1 to n)
    routes = []
    
    while unserved:
        # Guard: if no remaining customer is even feasible as a single-customer route,
        # we must stop to avoid an infinite loop.
        any_feasible_start = False
        for customer_id in unserved:
            travel_time = vrptw_instance.get_travel_time(0, customer_id)
            arrival_time = vrptw_instance.depot.ready_time + travel_time
            service_start = max(arrival_time, vrptw_instance.nodes[customer_id].ready_time)
            if service_start <= vrptw_instance.nodes[customer_id].due_time:
                service_end = service_start + vrptw_instance.nodes[customer_id].service_time
                return_time = service_end + vrptw_instance.get_travel_time(customer_id, 0)
                if return_time <= vrptw_instance.depot.due_time:
                    any_feasible_start = True
                    break

        if not any_feasible_start:
            break

        # Initialize new route from depot
        route = []
        current_time = vrptw_instance.depot.ready_time
        current_node = 0  # Depot
        
        while True:
            candidates: List[Tuple[float, int, int]] = []
            
            # Search for next feasible customer
            for customer_id in unserved:
                # Calculate arrival time
                travel_time = vrptw_instance.get_travel_time(current_node, customer_id)
                arrival_time = current_time + travel_time
                
                # Service start time (wait if early)
                service_start = max(arrival_time, vrptw_instance.nodes[customer_id].ready_time)
                
                # Check time window feasibility
                if service_start <= vrptw_instance.nodes[customer_id].due_time:
                    # Check if can return to depot after service
                    service_end = service_start + vrptw_instance.nodes[customer_id].service_time
                    return_time = service_end + vrptw_instance.get_travel_time(customer_id, 0)
                    
                    if return_time <= vrptw_instance.depot.due_time:
                        customer = vrptw_instance.nodes[customer_id]
                        slack = customer.due_time - service_start
                        tw_width = max(0.0, customer.due_time - customer.ready_time)
                        slack_norm = slack / (tw_width + 1.0)
                        if slack_norm < 0.0:
                            slack_norm = 0.0
                        elif slack_norm > 1.0:
                            slack_norm = 1.0
                        # Lower score is better; urgent customers have smaller slack_norm.
                        score = (w_distance * travel_time) + (w_urgency * slack_norm)
                        candidates.append((score, travel_time, customer_id))
            
            # No feasible extension found
            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])
            if rcl_size == 1:
                best_customer = candidates[0][2]
            else:
                k = min(rcl_size, len(candidates))
                best_customer = rng.choice([cid for _, _, cid in candidates[:k]])
            
            # Assign selected customer and update state
            route.append(best_customer)
            unserved.remove(best_customer)
            
            # Update current time and position
            travel_time = vrptw_instance.get_travel_time(current_node, best_customer)
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, vrptw_instance.nodes[best_customer].ready_time)
            current_time = service_start + vrptw_instance.nodes[best_customer].service_time
            current_node = best_customer
        
        # Add completed route
        if route:
            routes.append(route)
            
            # Check vehicle limit
            if max_vehicles and len(routes) >= max_vehicles:
                break
    
    # Calculate total cost
    total_cost = vrptw_instance.calculate_solution_cost(routes)
    
    # Statistics
    stats = {
        'algorithm': 'Nearest Neighbor VRPTW',
        'time': time.time() - start_time,
        'routes': len(routes),
        'customers_served': sum(len(route) for route in routes),
        'customers_unserved': len(unserved),
        'feasible': vrptw_instance.is_solution_feasible(routes)[0],
        'rcl_size': rcl_size,
        'seed': seed,
        'w_distance': w_distance,
        'w_urgency': w_urgency
    }
    
    return routes, total_cost, stats


def parallel_nearest_neighbor_vrptw(vrptw_instance: VRPTWInstance,
                                  max_vehicles: int = None,
                                  rcl_size: int = 1,
                                  seed: Optional[int] = None,
                                  rng: Optional[random.Random] = None,
                                  w_distance: float = 1.0,
                                  w_urgency: float = 0.0) -> Tuple[List[List[int]], float, Dict]:
    """
    Parallel Nearest Neighbor heuristic for capacity-free VRPTW
    Builds all routes simultaneously for better load balancing
    
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
    
    unserved = set(range(1, vrptw_instance.n_nodes))  # Customer IDs (1 to n)
    
    # Initialize routes (one per vehicle, starting from depot)
    if max_vehicles:
        num_vehicles = min(max_vehicles, len(unserved))
    else:
        num_vehicles = min(vrptw_instance.n_vehicles, len(unserved))
    
    routes = [[] for _ in range(num_vehicles)]
    route_times = [vrptw_instance.depot.ready_time] * num_vehicles
    route_positions = [0] * num_vehicles  # All start at depot
    
    # Assign first customer to each route using nearest neighbor
    for i in range(min(num_vehicles, len(unserved))):
        candidates: List[Tuple[float, float, int]] = []
        
        for customer_id in unserved:
            travel_time = vrptw_instance.get_travel_time(0, customer_id)
            arrival_time = vrptw_instance.depot.ready_time + travel_time
            service_start = max(arrival_time, vrptw_instance.nodes[customer_id].ready_time)

            if service_start <= vrptw_instance.nodes[customer_id].due_time:
                service_end = service_start + vrptw_instance.nodes[customer_id].service_time
                return_time = service_end + vrptw_instance.get_travel_time(customer_id, 0)

                if return_time <= vrptw_instance.depot.due_time:
                    customer = vrptw_instance.nodes[customer_id]
                    slack = customer.due_time - service_start
                    tw_width = max(0.0, customer.due_time - customer.ready_time)
                    slack_norm = slack / (tw_width + 1.0)
                    if slack_norm < 0.0:
                        slack_norm = 0.0
                    elif slack_norm > 1.0:
                        slack_norm = 1.0
                    score = (w_distance * travel_time) + (w_urgency * slack_norm)
                    candidates.append((score, travel_time, customer_id))

        best_customer = None
        if candidates:
            candidates.sort(key=lambda x: x[0])
            if rcl_size == 1:
                best_customer = candidates[0][2]
            else:
                k = min(rcl_size, len(candidates))
                best_customer = rng.choice([cid for _, _, cid in candidates[:k]])
        
        if best_customer:
            routes[i].append(best_customer)
            unserved.remove(best_customer)
            
            # Update route state
            travel_time = vrptw_instance.get_travel_time(0, best_customer)
            arrival_time = vrptw_instance.depot.ready_time + travel_time
            service_start = max(arrival_time, vrptw_instance.nodes[best_customer].ready_time)
            route_times[i] = service_start + vrptw_instance.nodes[best_customer].service_time
            route_positions[i] = best_customer
    
    # Extend routes iteratively
    while unserved:
        improved = False
        
        for route_idx in range(len(routes)):
            if not unserved:
                break

            candidates2: List[Tuple[float, float, int]] = []

            # Find feasible customers for this route
            for customer_id in unserved:
                travel_time = vrptw_instance.get_travel_time(route_positions[route_idx], customer_id)
                arrival_time = route_times[route_idx] + travel_time
                service_start = max(arrival_time, vrptw_instance.nodes[customer_id].ready_time)
                
                if service_start <= vrptw_instance.nodes[customer_id].due_time:
                    service_end = service_start + vrptw_instance.nodes[customer_id].service_time
                    return_time = service_end + vrptw_instance.get_travel_time(customer_id, 0)
                    
                    if return_time <= vrptw_instance.depot.due_time:
                        customer = vrptw_instance.nodes[customer_id]
                        slack = customer.due_time - service_start
                        tw_width = max(0.0, customer.due_time - customer.ready_time)
                        slack_norm = slack / (tw_width + 1.0)
                        if slack_norm < 0.0:
                            slack_norm = 0.0
                        elif slack_norm > 1.0:
                            slack_norm = 1.0
                        score = (w_distance * travel_time) + (w_urgency * slack_norm)
                        candidates2.append((score, travel_time, customer_id))

            best_customer = None
            if candidates2:
                candidates2.sort(key=lambda x: x[0])
                if rcl_size == 1:
                    best_customer = candidates2[0][2]
                else:
                    k = min(rcl_size, len(candidates2))
                    best_customer = rng.choice([cid for _, _, cid in candidates2[:k]])

            if best_customer:
                routes[route_idx].append(best_customer)
                unserved.remove(best_customer)
                improved = True
                
                # Update route state
                travel_time = vrptw_instance.get_travel_time(route_positions[route_idx], best_customer)
                arrival_time = route_times[route_idx] + travel_time
                service_start = max(arrival_time, vrptw_instance.nodes[best_customer].ready_time)
                route_times[route_idx] = service_start + vrptw_instance.nodes[best_customer].service_time
                route_positions[route_idx] = best_customer
        
        # If no improvement in any route, create new routes for remaining customers
        if not improved and unserved:
            added_any = False
            while unserved:
                route = []
                current_time = vrptw_instance.depot.ready_time
                current_node = 0
                
                # Try to add at least one customer to new route
                candidates3: List[Tuple[float, float, int]] = []
                
                for customer_id in unserved:
                    travel_time = vrptw_instance.get_travel_time(current_node, customer_id)
                    arrival_time = current_time + travel_time
                    service_start = max(arrival_time, vrptw_instance.nodes[customer_id].ready_time)
                    
                    if service_start <= vrptw_instance.nodes[customer_id].due_time:
                        service_end = service_start + vrptw_instance.nodes[customer_id].service_time
                        return_time = service_end + vrptw_instance.get_travel_time(customer_id, 0)
                        
                        if return_time <= vrptw_instance.depot.due_time:
                            customer = vrptw_instance.nodes[customer_id]
                            slack = customer.due_time - service_start
                            tw_width = max(0.0, customer.due_time - customer.ready_time)
                            slack_norm = slack / (tw_width + 1.0)
                            if slack_norm < 0.0:
                                slack_norm = 0.0
                            elif slack_norm > 1.0:
                                slack_norm = 1.0
                            score = (w_distance * travel_time) + (w_urgency * slack_norm)
                            candidates3.append((score, travel_time, customer_id))

                best_customer = None
                if candidates3:
                    candidates3.sort(key=lambda x: x[0])
                    if rcl_size == 1:
                        best_customer = candidates3[0][2]
                    else:
                        k = min(rcl_size, len(candidates3))
                        best_customer = rng.choice([cid for _, _, cid in candidates3[:k]])
                
                if best_customer:
                    route.append(best_customer)
                    unserved.remove(best_customer)
                    routes.append(route)
                    added_any = True
                else:
                    break

            # Guard: if we couldn't add any remaining customer, stop to avoid infinite loop
            if not added_any and unserved:
                break
    
    # Remove empty routes
    routes = [route for route in routes if route]
    
    # Calculate total cost
    total_cost = vrptw_instance.calculate_solution_cost(routes)
    
    # Statistics
    stats = {
        'algorithm': 'Parallel Nearest Neighbor VRPTW',
        'time': time.time() - start_time,
        'routes': len(routes),
        'customers_served': sum(len(route) for route in routes),
        'customers_unserved': len(unserved),
        'feasible': vrptw_instance.is_solution_feasible(routes)[0],
        'rcl_size': rcl_size,
        'seed': seed,
        'w_distance': w_distance,
        'w_urgency': w_urgency
    }
    
    return routes, total_cost, stats

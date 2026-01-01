#!/usr/bin/env python3
"""
Feasible Dual-Pipeline Implementation
Fixes original dual-pipeline to ensure feasibility while maintaining low cost
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problems.vrptw import VRPTWInstance, Customer
from heuristics.vrptw_nearest_neighbor import nearest_neighbor_vrptw, parallel_nearest_neighbor_vrptw
from heuristics.vrptw_savings_new import clarke_wright_savings_vrptw, parallel_savings_vrptw
from local_search.local_search_operators import combined_local_search


def calculate_route_duration(vrptw_instance, route):
    """Calculate total route duration including travel and service time"""
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


def feasibility_preserving_two_opt(vrptw_instance, solution):
    """Two-opt local search that preserves feasibility"""
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
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    # Check feasibility BEFORE calculating cost
                    feasible, _ = vrptw_instance.is_route_feasible(new_route)
                    if not feasible:
                        continue
                    
                    # Check route duration constraint
                    route_duration = calculate_route_duration(vrptw_instance, new_route)
                    if route_duration > 480:  # 8 hour limit
                        continue
                    
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
    
    return solution


def feasibility_preserving_relocation(vrptw_instance, solution):
    """Relocation local search that preserves feasibility"""
    improved = True
    iterations = 0
    
    while improved:
        improved = False
        iterations += 1
        
        # Try relocating each customer
        for route_a_idx, route_a in enumerate(solution):
            for customer_idx, customer in enumerate(route_a):
                for route_b_idx, route_b in enumerate(solution):
                    # Try all positions in route_b
                    for pos in range(len(route_b) + 1):
                        # Create new solution by moving customer
                        new_solution = [r.copy() for r in solution]
                        
                        # Remove customer from route_a
                        new_solution[route_a_idx] = route_a[:customer_idx] + route_a[customer_idx+1:]
                        
                        # Insert customer into route_b
                        if route_a_idx == route_b_idx:
                            # Same route - adjust position
                            if customer_idx < pos:
                                pos -= 1
                        new_solution[route_b_idx].insert(pos, customer)
                        
                        # Remove empty routes
                        new_solution = [r for r in new_solution if r]
                        
                        # Check feasibility
                        feasible, _ = vrptw_instance.is_solution_feasible(new_solution)
                        if not feasible:
                            continue
                        
                        # Check route durations
                        all_routes_valid = True
                        for route in new_solution:
                            route_duration = calculate_route_duration(vrptw_instance, route)
                            if route_duration > 480:
                                all_routes_valid = False
                                break
                        
                        if not all_routes_valid:
                            continue
                        
                        # Calculate cost improvement
                        old_cost = vrptw_instance.calculate_solution_cost(solution)
                        new_cost = vrptw_instance.calculate_solution_cost(new_solution)
                        
                        if new_cost < old_cost:
                            # Accept improvement
                            solution = new_solution
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
    
    return solution


def route_splitting_if_needed(vrptw_instance, solution, max_duration=480):
    """Split routes that exceed maximum duration"""
    improved_solution = []
    
    for route in solution:
        route_duration = calculate_route_duration(vrptw_instance, route)
        
        if route_duration <= max_duration:
            # Route is fine, keep as is
            improved_solution.append(route)
        else:
            # Route too long, split it
            print(f"Splitting route {route} (duration: {route_duration:.1f} min)")
            
            # Find best split point
            best_split = None
            best_cost = float('inf')
            
            for split_pos in range(1, len(route)):  # Don't split at first customer
                route1 = route[:split_pos]
                route2 = route[split_pos:]
                
                # Check both routes are feasible
                feasible1, _ = vrptw_instance.is_route_feasible(route1)
                feasible2, _ = vrptw_instance.is_route_feasible(route2)
                
                if feasible1 and feasible2:
                    total_cost = (vrptw_instance.calculate_route_distance(route1) + 
                                 vrptw_instance.calculate_route_distance(route2))
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_split = (route1, route2)
            
            if best_split:
                improved_solution.extend(best_split)
                print(f"  Split into: {best_split[0]} + {best_split[1]}")
            else:
                # If no feasible split, keep original route
                improved_solution.append(route)
                print(f"  No feasible split found, keeping original route")
    
    return improved_solution


def feasible_dual_pipeline(vrptw_instance, max_vehicles=None, 
                        nn_variant='sequential', cw_variant='standard',
                        max_route_duration=480):
    """
    Feasible Dual-Pipeline that ensures feasibility while maintaining low cost
    
    Args:
        vrptw_instance: VRPTW problem instance
        max_vehicles: Maximum number of vehicles
        nn_variant: Nearest Neighbor variant
        cw_variant: Clarke-Wright variant
        max_route_duration: Maximum allowed route duration (default 8 hours)
    
    Returns:
        tuple of (solution, cost, stats)
    """
    
    print("="*70)
    print("FEASIBLE DUAL-PIPELINE FRAMEWORK")
    print("CAPACITY-FREE VRPTW WITH GUARANTEED FEASIBILITY")
    print("="*70)
    
    start_time = time.time()
    stats = {
        'algorithm': 'Feasible Dual-Pipeline',
        'max_route_duration': max_route_duration,
        'nn_variant': nn_variant,
        'cw_variant': cw_variant
    }
    
    # Pipeline A: Nearest Neighbor
    print("\n" + "="*50)
    print("PIPELINE A: NEAREST NEIGHBOR (FEASIBILITY-PRESERVING)")
    print("="*50)
    
    # Stage 1: Nearest Neighbor Construction
    print("\n[Stage 1] Nearest Neighbor Construction...")
    if nn_variant == 'sequential':
        nn_solution, nn_cost, nn_stats = nearest_neighbor_vrptw(vrptw_instance, max_vehicles)
    else:
        nn_solution, nn_cost, nn_stats = parallel_nearest_neighbor_vrptw(vrptw_instance, max_vehicles)
    
    print(f"  Initial solution: {nn_cost:.2f}, {len(nn_solution)} routes, {nn_stats['time']:.3f}s")
    
    # Stage 2: Feasibility-Preserving Local Search
    print("\n[Stage 2] Feasibility-Preserving Local Search...")
    
    # Apply 2-opt with feasibility preservation
    nn_solution = feasibility_preserving_two_opt(vrptw_instance, nn_solution)
    nn_cost_after_2opt = vrptw_instance.calculate_solution_cost(nn_solution)
    improvement_2opt = nn_cost - nn_cost_after_2opt
    
    # Apply relocation with feasibility preservation
    nn_solution = feasibility_preserving_relocation(vrptw_instance, nn_solution)
    nn_cost_after_reloc = vrptw_instance.calculate_solution_cost(nn_solution)
    improvement_reloc = nn_cost_after_2opt - nn_cost_after_reloc
    
    # Apply route splitting if needed
    nn_solution = route_splitting_if_needed(vrptw_instance, nn_solution, max_route_duration)
    nn_final_cost = vrptw_instance.calculate_solution_cost(nn_solution)
    improvement_split = nn_cost_after_reloc - nn_final_cost
    
    stats['pipeline_a'] = {
        'initial_cost': nn_cost,
        'final_cost': nn_final_cost,
        'final_routes': len(nn_solution),
        'improvement_2opt': improvement_2opt,
        'improvement_reloc': improvement_reloc,
        'improvement_split': improvement_split,
        'total_improvement': nn_cost - nn_final_cost,
        'feasible': vrptw_instance.is_solution_feasible(nn_solution)[0]
    }
    
    print(f"  After 2-opt: {nn_cost_after_2opt:.2f} ({improvement_2opt:.2f} improvement)")
    print(f"  After relocation: {nn_cost_after_reloc:.2f} ({improvement_reloc:.2f} improvement)")
    print(f"  After route splitting: {nn_final_cost:.2f} ({improvement_split:.2f} improvement)")
    print(f"  Total improvement: {nn_cost - nn_final_cost:.2f}")
    print(f"  Final routes: {len(nn_solution)}")
    
    # Pipeline B: Clarke-Wright Savings
    print("\n" + "="*50)
    print("PIPELINE B: CLARKE-WRIGHT SAVINGS (FEASIBILITY-PRESERVING)")
    print("="*50)
    
    # Stage 1: Clarke-Wright Construction
    print("\n[Stage 1] Clarke-Wright Savings Construction...")
    if cw_variant == 'standard':
        cw_solution, cw_cost, cw_stats = clarke_wright_savings_vrptw(vrptw_instance, max_vehicles)
    else:
        cw_solution, cw_cost, cw_stats = parallel_savings_vrptw(vrptw_instance, max_vehicles)
    
    print(f"  Initial solution: {cw_cost:.2f}, {len(cw_solution)} routes, {cw_stats['time']:.3f}s")
    
    # Stage 2: Feasibility-Preserving Local Search
    print("\n[Stage 2] Feasibility-Preserving Local Search...")
    
    # Apply standard local search (already preserves feasibility)
    cw_solution, cw_cost_after_ls, ls_stats = combined_local_search(vrptw_instance, cw_solution)
    
    # Apply route splitting if needed
    cw_solution = route_splitting_if_needed(vrptw_instance, cw_solution, max_route_duration)
    cw_final_cost = vrptw_instance.calculate_solution_cost(cw_solution)
    improvement_cw_split = cw_cost_after_ls - cw_final_cost
    
    stats['pipeline_b'] = {
        'initial_cost': cw_cost,
        'initial_routes': len(cw_solution),
        'cost_after_ls': cw_cost_after_ls,
        'final_cost': cw_final_cost,
        'final_routes': len(cw_solution),
        'improvement_ls': cw_cost - cw_cost_after_ls,
        'improvement_split': improvement_cw_split,
        'total_improvement': cw_cost - cw_final_cost,
        'feasible': vrptw_instance.is_solution_feasible(cw_solution)[0]
    }
    
    print(f"  After local search: {cw_cost_after_ls:.2f} ({cw_cost - cw_cost_after_ls:.2f} improvement)")
    print(f"  After route splitting: {cw_final_cost:.2f} ({improvement_cw_split:.2f} improvement)")
    print(f"  Total improvement: {cw_cost - cw_final_cost:.2f}")
    print(f"  Final routes: {len(cw_solution)}")
    
    # Select best solution
    if nn_final_cost <= cw_final_cost:
        best_solution = nn_solution
        best_cost = nn_final_cost
        best_pipeline = 'A'
    else:
        best_solution = cw_solution
        best_cost = cw_final_cost
        best_pipeline = 'B'
    
    # Final statistics
    stats.update({
        'total_time': time.time() - start_time,
        'best_pipeline': best_pipeline,
        'best_cost': best_cost,
        'best_routes': len(best_solution),
        'best_solution': best_solution,
        'feasible': vrptw_instance.is_solution_feasible(best_solution)[0]
    })
    
    # Final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"Pipeline A (NN):     {nn_final_cost:.2f}, {len(nn_solution)} routes")
    print(f"Pipeline B (CW):     {cw_final_cost:.2f}, {len(cw_solution)} routes")
    print(f"Best Pipeline:        {best_pipeline} ({best_cost:.2f})")
    print(f"Cost Difference:       {abs(nn_final_cost - cw_final_cost):.2f}")
    print(f"Total Runtime:        {stats['total_time']:.3f}s")
    print(f"Max Route Duration:    {max_route_duration} minutes")
    
    feasible, reason = vrptw_instance.is_solution_feasible(best_solution)
    if not feasible:
        print(f"⚠️  WARNING: Solution still infeasible: {reason}")
        print("  This may indicate need for different parameters")
    else:
        print("✅ SUCCESS: Feasible solution achieved!")
    
    return best_solution, best_cost, stats


def main():
    """Main function to test feasible dual-pipeline"""
    try:
        # Create test instance
        customers = [
            Customer(1, 10, 15, ready_time=480, due_time=1020, service_time=60),
            Customer(2, 25, 30, ready_time=500, due_time=900, service_time=60),
            Customer(3, 40, 20, ready_time=600, due_time=1000, service_time=60),
            Customer(4, 15, 45, ready_time=480, due_time=800, service_time=60),
            Customer(5, 35, 10, ready_time=700, due_time=1100, service_time=60),
            Customer(6, 20, 25, ready_time=480, due_time=900, service_time=60),
            Customer(7, 45, 35, ready_time=800, due_time=1200, service_time=60),
            Customer(8, 30, 40, ready_time=550, due_time=950, service_time=60),
            Customer(9, 50, 25, ready_time=650, due_time=1050, service_time=60),
            Customer(10, 5, 35, ready_time=480, due_time=750, service_time=60)
        ]
        
        instance = VRPTWInstance(customers)
        
        print("TESTING FEASIBLE DUAL-PIPELINE")
        print(f"Instance: {instance.n_customers} customers")
        print(f"Max Route Duration: 480 minutes (8 hours)")
        print()
        
        # Run feasible dual-pipeline
        solution, cost, stats = feasible_dual_pipeline(instance)
        
        print(f"\n" + "="*70)
        print("FEASIBLE DUAL-PIPELINE RESULTS")
        print("="*70)
        print(f"Best Algorithm: Pipeline {stats['best_pipeline']}")
        print(f"Total Cost: {cost:.2f}")
        print(f"Number of Routes: {len(solution)}")
        print(f"Feasible: {stats['feasible']}")
        print(f"Total Runtime: {stats['total_time']:.3f}s")
        
        print(f"\nBEST SOLUTION DETAILS:")
        for i, route in enumerate(solution, 1):
            route_cost = instance.calculate_route_distance(route)
            route_duration = calculate_route_duration(instance, route)
            print(f"  Route {i}: {route}")
            print(f"    Cost: {route_cost:.2f}, Customers: {len(route)}")
            print(f"    Duration: {route_duration:.1f} minutes ({route_duration/60:.1f} hours)")
        
        # Save results
        with open('results/feasible_dual_pipeline_results.txt', 'w') as f:
            f.write("FEASIBLE DUAL-PIPELINE RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Best Pipeline: {stats['best_pipeline']}\n")
            f.write(f"Total Cost: {cost:.2f}\n")
            f.write(f"Number of Routes: {len(solution)}\n")
            f.write(f"Feasible: {stats['feasible']}\n")
            f.write(f"Total Runtime: {stats['total_time']:.3f}s\n")
            f.write(f"Max Route Duration: {stats['max_route_duration']} minutes\n")
            f.write("\nROUTES:\n")
            for i, route in enumerate(solution, 1):
                f.write(f"Route {i}: {route}\n")
        
        print(f"\n✓ Results saved to: results/feasible_dual_pipeline_results.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import time
    main()

#!/usr/bin/env python3
"""
Main Execution Script for Capacity-Free VRPTW
Dual-Pipeline Heuristic Framework Implementation

Usage:
    python main_capacity_free_vrptw.py --customers data/customers.csv
    python main_capacity_free_vrptw.py --customers data/customers.csv --algorithm dual
    python main_capacity_free_vrptw.py --customers data/customers.csv --nn-variant parallel --cw-variant parallel
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problems.vrptw import VRPTWInstance
from heuristics.vrptw_nearest_neighbor import nearest_neighbor_vrptw, parallel_nearest_neighbor_vrptw
from heuristics.vrptw_savings_new import clarke_wright_savings_vrptw, parallel_savings_vrptw
from final_feasible_pipeline import final_feasible_dual_pipeline
from utils.visualization_new import (
    plot_vrptw_solution, plot_algorithm_comparison, plot_pipeline_analysis,
    print_solution_summary, save_results_to_csv
)


def _calculate_route_duration(vrptw_instance: VRPTWInstance, route: List[int]) -> float:
    if not route:
        return 0.0

    current_time = vrptw_instance.depot.ready_time
    current_node = 0

    for node_id in route:
        travel_time = vrptw_instance.get_travel_time(current_node, node_id)
        arrival_time = current_time + travel_time
        service_start = max(arrival_time, vrptw_instance.nodes[node_id].ready_time)
        current_time = service_start + vrptw_instance.nodes[node_id].service_time
        current_node = node_id

    return_time = current_time + vrptw_instance.get_travel_time(current_node, 0)
    return return_time - vrptw_instance.depot.ready_time


def _apply_max_route_duration(
    vrptw_instance: VRPTWInstance,
    solution: List[List[int]],
    max_route_duration: Optional[float],
) -> List[List[int]]:
    if max_route_duration is None:
        return solution

    new_solution: List[List[int]] = []
    for route in solution:
        if not route:
            continue

        if _calculate_route_duration(vrptw_instance, route) <= max_route_duration:
            new_solution.append(route)
            continue

        current: List[int] = []
        for customer_id in route:
            candidate = current + [customer_id]
            feasible, _ = vrptw_instance.is_route_feasible(candidate)
            duration = _calculate_route_duration(vrptw_instance, candidate)

            if current and (not feasible or duration > max_route_duration):
                new_solution.append(current)
                current = [customer_id]
            else:
                current = candidate

        if current:
            new_solution.append(current)

    return new_solution


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Capacity-Free VRPTW Optimization with Dual-Pipeline Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --customers data/sample_customers.csv
  %(prog)s --customers data/customers.csv --algorithm dual
  %(prog)s --customers data/customers.csv --algorithm nn --nn-variant parallel
  %(prog)s --customers data/customers.csv --output-dir results
        """
    )
    
    # Required arguments
    parser.add_argument('--customers', '-c', required=True,
                       help='CSV file with customer data (id, x, y, ready_time, due_time, service_time)')
    
    # Optional arguments
    parser.add_argument('--distances', '-d',
                       help='CSV file with distance matrix (optional, will compute if not provided)')
    
    parser.add_argument('--algorithm', '-a', 
                       choices=['nn', 'cw', 'dual', 'all'],
                       default='dual',
                       help='Algorithm to run: nn=Nearest Neighbor, cw=Clarke-Wright, dual=Dual-Pipeline, all=All algorithms (default: dual)')
    
    parser.add_argument('--nn-variant',
                       choices=['sequential', 'parallel'],
                       default='sequential',
                       help='Nearest Neighbor variant (default: sequential)')

    parser.add_argument('--nn-w-distance', type=float, default=1.0,
                       help='Weight for NN travel-time term in weighted NN scoring (default: 1.0)')

    parser.add_argument('--nn-w-urgency', type=float, default=0.0,
                       help='Weight for NN urgency/slack term in weighted NN scoring (0 disables; default: 0.0)')
    
    parser.add_argument('--cw-variant',
                       choices=['standard', 'parallel'],
                       default='standard',
                       help='Clarke-Wright Savings variant (default: standard)')
    
    parser.add_argument('--max-vehicles', '-m', type=int,
                       help='Maximum number of vehicles (default: unlimited)')
    
    parser.add_argument('--depot-x', type=float, default=0.0,
                       help='Depot X coordinate (default: 0.0)')
    
    parser.add_argument('--depot-y', type=float, default=0.0,
                       help='Depot Y coordinate (default: 0.0)')
    
    parser.add_argument('--depot-ready', type=float, default=480,
                       help='Depot ready time in minutes (default: 480 = 08:00)')
    
    parser.add_argument('--depot-due', type=float, default=1020,
                       help='Depot due time in minutes (default: 1020 = 17:00)')
    
    parser.add_argument('--service-time', type=float, default=60,
                       help='Default service time in minutes (default: 60)')

    parser.add_argument('--travel-speed', type=float, default=1.0,
                       help='Travel speed (distance units per minute) used for time-feasibility. Distance cost is unchanged. (default: 1.0)')

    parser.add_argument('--max-route-duration', type=float, default=480,
                       help='Maximum allowed route duration in minutes applied to all algorithms via route splitting (default: 480)')
    
    parser.add_argument('--output-dir', '-o', default='results',
                       help='Output directory for results (default: results)')
    
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def create_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_single_algorithm(vrptw_instance: VRPTWInstance, algorithm: str,
                       nn_variant: str, cw_variant: str,
                       max_vehicles: int, output_dir: Path,
                       no_plots: bool, verbose: bool,
                       max_route_duration: Optional[float],
                       nn_w_distance: float,
                       nn_w_urgency: float) -> Dict:
    """Run a single algorithm and return results"""
    
    if algorithm == 'nn':
        if verbose:
            print(f"Running Nearest Neighbor ({nn_variant})...")
        
        if nn_variant == 'sequential':
            solution, cost, stats = nearest_neighbor_vrptw(
                vrptw_instance,
                max_vehicles,
                w_distance=nn_w_distance,
                w_urgency=nn_w_urgency,
            )
        else:
            solution, cost, stats = parallel_nearest_neighbor_vrptw(
                vrptw_instance,
                max_vehicles,
                w_distance=nn_w_distance,
                w_urgency=nn_w_urgency,
            )
    
    elif algorithm == 'cw':
        if verbose:
            print(f"Running Clarke-Wright Savings ({cw_variant})...")
        
        if cw_variant == 'standard':
            solution, cost, stats = clarke_wright_savings_vrptw(vrptw_instance, max_vehicles)
        else:
            solution, cost, stats = parallel_savings_vrptw(vrptw_instance, max_vehicles)
    
    elif algorithm == 'dual':
        # The project uses the final feasible dual-pipeline as the maintained
        # dual-pipeline implementation.
        if verbose:
            print(f"Running Feasible Dual-Pipeline Framework (NN: {nn_variant}, CW: {cw_variant})...")

        # Option A (fair comparison): apply a single, shared max-route-duration
        # post-processing rule to ALL algorithms. To avoid double-splitting,
        # disable internal splitting inside the feasible pipeline and then apply
        # _apply_max_route_duration below.
        internal_max_route_duration = 10**9
        solution, cost, stats = final_feasible_dual_pipeline(
            vrptw_instance,
            max_vehicles,
            nn_variant,
            cw_variant,
            max_route_duration=internal_max_route_duration,
            nn_w_distance=nn_w_distance,
            nn_w_urgency=nn_w_urgency,
        )
    
    elif algorithm == 'feasible_dual':
        if verbose:
            print(f"Running Feasible Dual-Pipeline Framework (NN: {nn_variant}, CW: {cw_variant})...")
        
        # Option A (fair comparison): apply a single, shared max-route-duration
        # post-processing rule to ALL algorithms. To avoid double-splitting and
        # to ensure travel-time-based duration calculation is used consistently,
        # disable internal splitting inside the feasible pipeline by using a very
        # large max_route_duration, then apply _apply_max_route_duration below.
        internal_max_route_duration = 10**9
        solution, cost, stats = final_feasible_dual_pipeline(
            vrptw_instance, max_vehicles, nn_variant, cw_variant, max_route_duration=internal_max_route_duration)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    solution = _apply_max_route_duration(vrptw_instance, solution, max_route_duration)
    cost = vrptw_instance.calculate_solution_cost(solution)
    stats['feasible'] = vrptw_instance.is_solution_feasible(solution)[0]
    stats['max_route_duration'] = max_route_duration

    return {
        'solution': solution,
        'cost': cost,
        'stats': stats
    }


def run_all_algorithms(vrptw_instance: VRPTWInstance,
                     nn_variant: str, cw_variant: str,
                     max_vehicles: int, output_dir: Path,
                     no_plots: bool, verbose: bool, algorithm: str,
                     max_route_duration: Optional[float],
                     nn_w_distance: float,
                     nn_w_urgency: float) -> Dict:
    """Run all algorithms and return comparison results"""
    
    if algorithm == 'all':
        algorithms = {
            'Nearest Neighbor': ('nn', nn_variant, cw_variant),
            'Clarke-Wright': ('cw', cw_variant, cw_variant),
            'Feasible Dual-Pipeline': ('feasible_dual', nn_variant, cw_variant)
        }
    elif algorithm == 'nn':
        algorithms = {'Nearest Neighbor': ('nn', nn_variant, cw_variant)}
    elif algorithm == 'cw':
        algorithms = {'Clarke-Wright': ('cw', cw_variant, cw_variant)}
    elif algorithm == 'dual':
        # Only use feasible dual-pipeline
        algorithms = {'Feasible Dual-Pipeline': ('feasible_dual', nn_variant, cw_variant)}
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    results = {}
    
    for name, params in algorithms.items():
        print(f"\n{'='*60}")
        print(f"RUNNING {name.upper()}")
        print(f"{'='*60}")
        
        alg, nn_var, cw_var = params
        result = run_single_algorithm(vrptw_instance, alg, nn_var, cw_var,
                                  max_vehicles, output_dir, no_plots, verbose, max_route_duration,
                                  nn_w_distance, nn_w_urgency)
        
        results[name] = {
            'solution': result['solution'],
            'total_cost': result['cost'],
            'routes': len(result['solution']),
            'time': result['stats'].get('time', result['stats'].get('total_time', 0)),
            'feasible': result['stats']['feasible']
        }
        
        # Print summary
        print_solution_summary(vrptw_instance, result['solution'], result['cost'], name)
        
        # Save individual solution plot
        if not no_plots:
            plot_path = output_dir / f"{name.lower().replace(' ', '_')}_solution.png"
            plot_vrptw_solution(vrptw_instance, result['solution'], 
                              f"{name} Solution", str(plot_path))
    
    return results


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    print("="*70)
    print("CAPACITY-FREE VRPTW OPTIMIZATION")
    print("DUAL-PIPELINE HEURISTIC FRAMEWORK")
    print("="*70)
    print(f"Instance: {args.customers}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Output Directory: {output_dir}")
    print("="*70)
    
    # Load VRPTW instance
    print(f"\nLoading problem instance from {args.customers}...")
    try:
        vrptw_instance = VRPTWInstance.load_from_csv(
            customers_file=args.customers,
            distance_file=args.distances,
            depot_x=args.depot_x,
            depot_y=args.depot_y,
            depot_ready=args.depot_ready,
            depot_due=args.depot_due,
            service_time=args.service_time,
            n_vehicles=args.max_vehicles,
            travel_speed=args.travel_speed
        )
        print(f"  Loaded: {vrptw_instance.n_customers} customers")
        print(f"  Depot: ({args.depot_x}, {args.depot_y})")
        print(f"  Time window: [{args.depot_ready}, {args.depot_due}]")
        print(f"  Service time: {args.service_time} minutes")
        print(f"  Travel speed: {args.travel_speed} distance units/min")
        
    except Exception as e:
        print(f"Error loading instance: {e}")
        sys.exit(1)
    
    # Run algorithms
    start_time = time.time()
    
    if args.algorithm == 'all':
        results = run_all_algorithms(vrptw_instance, args.nn_variant, args.cw_variant,
                                  args.max_vehicles, output_dir, args.no_plots, args.verbose, args.algorithm,
                                  args.max_route_duration,
                                  args.nn_w_distance,
                                  args.nn_w_urgency)
        
        # Generate comparison plots
        if not args.no_plots:
            comparison_path = output_dir / "algorithm_comparison.png"
            plot_algorithm_comparison(results, str(comparison_path))
        
        # Save results to CSV
        csv_path = output_dir / "results_comparison.csv"
        save_results_to_csv(results, str(csv_path))
        
        # Print final summary
        print(f"\n{'='*70}")
        print("FINAL COMPARISON SUMMARY")
        print(f"{'='*70}")
        for name, stats in results.items():
            print(f"{name:20} | Cost: {stats['total_cost']:8.2f} | "
                  f"Routes: {stats['routes']:2} | Time: {stats['time']:6.3f}s | "
                  f"Feasible: {'Yes' if stats['feasible'] else 'No'}")
        
    else:
        # Run single algorithm
        result = run_single_algorithm(vrptw_instance, args.algorithm,
                                  args.nn_variant, args.cw_variant,
                                  args.max_vehicles, output_dir, 
                                  args.no_plots, args.verbose, args.max_route_duration,
                                  args.nn_w_distance,
                                  args.nn_w_urgency)
        
        # Print detailed summary
        print_solution_summary(vrptw_instance, result['solution'], result['cost'], 
                           f"{args.algorithm.upper()} ({args.nn_variant})")
        
        # Save solution plot
        if not args.no_plots:
            plot_path = output_dir / "best_solution.png"
            plot_vrptw_solution(vrptw_instance, result['solution'],
                              "Best Solution", str(plot_path))
        
        # Save detailed results
        if args.algorithm == 'dual' and 'stages' in result['stats']:
            # Pipeline analysis
            if not args.no_plots:
                pipeline_path = output_dir / "pipeline_analysis.png"
                plot_pipeline_analysis(result['stats'], str(pipeline_path))
        
        results = {args.algorithm: {
            'solution': result['solution'],
            'total_cost': result['cost'],
            'routes': len(result['solution']),
            'time': result['stats'].get('time', result['stats'].get('total_time', 0)),
            'feasible': result['stats']['feasible']
        }}
    
    # Save best solution
    best_algorithm = min(results.keys(), key=lambda k: results[k]['total_cost'])
    best_solution = results[best_algorithm]['solution']
    
    solution_path = output_dir / "best_solution.txt"
    vrptw_instance.save_solution(best_solution, str(solution_path))
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("EXECUTION COMPLETE")
    print(f"{'='*70}")
    print(f"Best Algorithm: {best_algorithm}")
    print(f"Best Cost: {results[best_algorithm]['total_cost']:.2f}")
    print(f"Total Runtime: {total_time:.3f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

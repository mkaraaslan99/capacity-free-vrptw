"""
Main execution script for VRPTW optimization project
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problems.vrptw import VRPTWInstance
from heuristics.vrptw_nearest_neighbor import nearest_neighbor_vrptw, parallel_nearest_neighbor_vrptw
from heuristics.vrptw_savings import clarke_wright_savings_vrptw, parallel_savings_vrptw
from metaheuristics.genetic_algorithm_vrptw import GeneticAlgorithmVRPTW
from metaheuristics.simulated_annealing_vrptw import SimulatedAnnealingVRPTW
from utils.visualization import (plot_vrptw_solution, plot_algorithm_comparison, 
                                 print_solution_summary, plot_convergence)


def run_all_algorithms(vrptw_instance, n_vehicles=None):
    """
    Run all implemented algorithms on the instance
    
    Args:
        vrptw_instance: VRPTWInstance object
        n_vehicles: maximum number of vehicles
        
    Returns:
        dict of results
    """
    results = {}
    
    print("\n" + "="*70)
    print("RUNNING ALL ALGORITHMS")
    print("="*70)
    
    # 1. Nearest Neighbor
    print("\n[1/6] Running Nearest Neighbor Heuristic...")
    solution, cost, stats = nearest_neighbor_vrptw(vrptw_instance, n_vehicles)
    results['Nearest Neighbor'] = {'solution': solution, 'cost': cost, 'stats': stats}
    print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
    
    # 2. Parallel Nearest Neighbor
    print("\n[2/6] Running Parallel Nearest Neighbor...")
    solution, cost, stats = parallel_nearest_neighbor_vrptw(vrptw_instance, n_vehicles)
    results['Parallel NN'] = {'solution': solution, 'cost': cost, 'stats': stats}
    print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
    
    # 3. Clarke-Wright Savings
    print("\n[3/6] Running Clarke-Wright Savings Algorithm...")
    solution, cost, stats = clarke_wright_savings_vrptw(vrptw_instance, n_vehicles)
    results['Clarke-Wright'] = {'solution': solution, 'cost': cost, 'stats': stats}
    print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
    
    # 4. Parallel Savings
    print("\n[4/6] Running Parallel Savings Algorithm...")
    solution, cost, stats = parallel_savings_vrptw(vrptw_instance, n_vehicles)
    results['Parallel Savings'] = {'solution': solution, 'cost': cost, 'stats': stats}
    print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
    
    # 5. Genetic Algorithm
    print("\n[5/6] Running Genetic Algorithm...")
    # Optimized parameters for 50 customers: smaller population and fewer generations
    ga = GeneticAlgorithmVRPTW(vrptw_instance, population_size=20, n_generations=30)
    solution, cost, stats = ga.solve()
    results['Genetic Algorithm'] = {'solution': solution, 'cost': cost, 'stats': stats}
    print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
    
    # 6. Simulated Annealing
    print("\n[6/6] Running Simulated Annealing...")
    # Use best heuristic solution as initial solution
    best_heuristic = min(results.values(), key=lambda x: x['cost'])
    # Optimized parameters: faster cooling and fewer iterations per temp
    sa = SimulatedAnnealingVRPTW(vrptw_instance, initial_temp=500, cooling_rate=0.95, iterations_per_temp=50)
    solution, cost, stats = sa.solve(initial_solution=best_heuristic['solution'])
    results['Simulated Annealing'] = {'solution': solution, 'cost': cost, 'stats': stats}
    print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='VRPTW Optimization with Heuristics and Metaheuristics')
    
    parser.add_argument('--customers', type=str, required=True,
                       help='Path to customers CSV file')
    parser.add_argument('--distances', type=str, default=None,
                       help='Path to distance matrix CSV file (optional)')
    parser.add_argument('--depot-x', type=float, default=0.0,
                       help='Depot X coordinate (default: 0.0)')
    parser.add_argument('--depot-y', type=float, default=0.0,
                       help='Depot Y coordinate (default: 0.0)')
    parser.add_argument('--depot-open', type=float, default=480.0,
                       help='Depot opening time in minutes from midnight (default: 480 = 8:00)')
    parser.add_argument('--depot-close', type=float, default=1020.0,
                       help='Depot closing time in minutes from midnight (default: 1020 = 17:00)')
    parser.add_argument('--service-time', type=float, default=60.0,
                       help='Service time per customer in minutes (default: 60)')
    parser.add_argument('--capacity', type=float, default=float('inf'),
                       help='Vehicle capacity (default: infinite)')
    parser.add_argument('--n-vehicles', type=int, default=None,
                       help='Number of vehicles (default: unlimited)')
    parser.add_argument('--algorithm', type=str, default='all',
                       choices=['all', 'nn', 'parallel-nn', 'savings', 'parallel-savings', 'ga', 'sa'],
                       help='Algorithm to run (default: all)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: run only heuristics (skip metaheuristics)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load instance
    print("\n" + "="*70)
    print("LOADING VRPTW INSTANCE")
    print("="*70)
    print(f"Customers file: {args.customers}")
    print(f"Depot location: ({args.depot_x}, {args.depot_y})")
    print(f"Operating hours: {args.depot_open/60:.1f}:00 - {args.depot_close/60:.1f}:00")
    print(f"Service time: {args.service_time} minutes")
    
    vrptw_instance = VRPTWInstance.load_from_csv(
        customers_file=args.customers,
        distance_file=args.distances,
        depot_x=args.depot_x,
        depot_y=args.depot_y,
        depot_ready=args.depot_open,
        depot_due=args.depot_close,
        service_time=args.service_time,
        vehicle_capacity=args.capacity,
        n_vehicles=args.n_vehicles
    )
    
    print(f"\nLoaded {vrptw_instance.n_customers} customers")
    
    # Run algorithms
    if args.algorithm == 'all':
        if args.fast:
            # Fast mode: only heuristics
            print("\n⚡ FAST MODE: Running only heuristics (skipping metaheuristics)")
            results = {}
            
            print("\n[1/4] Running Nearest Neighbor Heuristic...")
            solution, cost, stats = nearest_neighbor_vrptw(vrptw_instance, args.n_vehicles)
            results['Nearest Neighbor'] = {'solution': solution, 'cost': cost, 'stats': stats}
            print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
            
            print("\n[2/4] Running Parallel Nearest Neighbor...")
            solution, cost, stats = parallel_nearest_neighbor_vrptw(vrptw_instance, args.n_vehicles)
            results['Parallel NN'] = {'solution': solution, 'cost': cost, 'stats': stats}
            print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
            
            print("\n[3/4] Running Clarke-Wright Savings Algorithm...")
            solution, cost, stats = clarke_wright_savings_vrptw(vrptw_instance, args.n_vehicles)
            results['Clarke-Wright'] = {'solution': solution, 'cost': cost, 'stats': stats}
            print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
            
            print("\n[4/4] Running Parallel Savings Algorithm...")
            solution, cost, stats = parallel_savings_vrptw(vrptw_instance, args.n_vehicles)
            results['Parallel Savings'] = {'solution': solution, 'cost': cost, 'stats': stats}
            print(f"  Cost: {cost:.2f}, Routes: {len(solution)}, Time: {stats['time']:.3f}s")
        else:
            results = run_all_algorithms(vrptw_instance, args.n_vehicles)
        
        # Separate feasible and infeasible solutions
        feasible_results = {name: res for name, res in results.items() if res['stats'].get('feasible', False)}
        infeasible_results = {name: res for name, res in results.items() if not res['stats'].get('feasible', False)}
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        if feasible_results:
            print(f"\n✅ FEASIBLE SOLUTIONS ({len(feasible_results)}/{len(results)}):")
            print("-" * 70)
            for name, res in sorted(feasible_results.items(), key=lambda x: x[1]['cost']):
                print(f"  {name:30s} | Cost: {res['cost']:10.2f} | Routes: {len(res['solution']):3d} | Time: {res['stats']['time']:6.3f}s")
        
        if infeasible_results:
            print(f"\n❌ INFEASIBLE SOLUTIONS ({len(infeasible_results)}/{len(results)}):")
            print("-" * 70)
            for name, res in infeasible_results.items():
                reason = res['stats'].get('feasibility_reason', 'Unknown')
                print(f"  {name:30s} | Cost: {res['cost']:10.2f} | Routes: {len(res['solution']):3d}")
                print(f"    Reason: {reason[:80]}")
        
        # Find best feasible solution
        if feasible_results:
            best_name = min(feasible_results.keys(), key=lambda k: feasible_results[k]['cost'])
            best_result = feasible_results[best_name]
            
            print("\n" + "="*70)
            print("BEST FEASIBLE SOLUTION")
            print("="*70)
            print(f"Algorithm: {best_name}")
            print(f"Cost: {best_result['cost']:.2f}")
            print(f"Routes: {len(best_result['solution'])}")
        else:
            print("\n" + "="*70)
            print("⚠️  WARNING: NO FEASIBLE SOLUTIONS FOUND")
            print("="*70)
            print("All algorithms produced infeasible solutions.")
            print("Consider: increasing vehicles, extending time windows, or reducing service time.")
            # Use best infeasible solution
            best_name = min(results.keys(), key=lambda k: results[k]['cost'])
            best_result = results[best_name]
            print(f"\nShowing best infeasible solution: {best_name}")
        
        # Print detailed summary
        print_solution_summary(vrptw_instance, best_result['solution'], best_result['stats'])
        
        # Save all results to comprehensive file
        all_results_file = os.path.join(args.output_dir, 'all_results.txt')
        with open(all_results_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ALL ALGORITHM RESULTS\n")
            f.write("="*70 + "\n")
            f.write(f"\nInstance: {vrptw_instance.name}\n")
            f.write(f"Total Customers: {vrptw_instance.n_customers}\n")
            f.write(f"Depot: ({vrptw_instance.depot.x}, {vrptw_instance.depot.y})\n")
            f.write(f"Time Window: {vrptw_instance.depot.ready_time} - {vrptw_instance.depot.due_time} minutes\n")
            f.write(f"Max Vehicles: {args.n_vehicles if args.n_vehicles else 'Unlimited'}\n")
            f.write("\n")
            
            # Feasible solutions
            if feasible_results:
                f.write("\n" + "="*70 + "\n")
                f.write(f"FEASIBLE SOLUTIONS ({len(feasible_results)}/{len(results)})\n")
                f.write("="*70 + "\n\n")
                
                for name, res in sorted(feasible_results.items(), key=lambda x: x[1]['cost']):
                    f.write("-" * 70 + "\n")
                    f.write(f"Algorithm: {name}\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Total Cost: {res['cost']:.2f}\n")
                    f.write(f"Number of Routes: {len(res['solution'])}\n")
                    f.write(f"Computation Time: {res['stats']['time']:.3f} seconds\n")
                    f.write(f"Feasible: Yes\n")
                    
                    # Route details
                    f.write(f"\nRoute Details:\n")
                    for i, route in enumerate(res['solution'], 1):
                        if route:
                            distance = vrptw_instance.calculate_route_distance(route)
                            demand = sum(vrptw_instance.nodes[c].demand for c in route)
                            f.write(f"  Route {i}: 0 -> {' -> '.join(map(str, route))} -> 0\n")
                            f.write(f"    Distance: {distance:.2f}, Customers: {len(route)}, Demand: {demand:.2f}\n")
                    f.write("\n")
            
            # Infeasible solutions
            if infeasible_results:
                f.write("\n" + "="*70 + "\n")
                f.write(f"INFEASIBLE SOLUTIONS ({len(infeasible_results)}/{len(results)})\n")
                f.write("="*70 + "\n\n")
                
                for name, res in infeasible_results.items():
                    f.write("-" * 70 + "\n")
                    f.write(f"Algorithm: {name}\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Total Cost: {res['cost']:.2f}\n")
                    f.write(f"Number of Routes: {len(res['solution'])}\n")
                    f.write(f"Computation Time: {res['stats']['time']:.3f} seconds\n")
                    f.write(f"Feasible: No\n")
                    reason = res['stats'].get('feasibility_reason', 'Unknown')
                    f.write(f"Reason: {reason}\n\n")
        
        print(f"\nAll results saved to: {all_results_file}")
        
        # Save best solution
        output_file = os.path.join(args.output_dir, 'best_solution.txt')
        vrptw_instance.save_solution(best_result['solution'], output_file)
        print(f"Best solution saved to: {output_file}")
        
        # Visualizations
        if not args.no_plot:
            print("\nGenerating visualizations...")
            
            # Plot best solution
            plot_path = os.path.join(args.output_dir, 'best_solution.png')
            plot_vrptw_solution(vrptw_instance, best_result['solution'], 
                              title=f"Best Solution - {best_name}", 
                              save_path=plot_path, show=False)
            print(f"  Solution plot saved to: {plot_path}")
            
            # Plot comparison
            stats_list = [r['stats'] for r in results.values()]
            comparison_path = os.path.join(args.output_dir, 'algorithm_comparison.png')
            plot_algorithm_comparison(stats_list, save_path=comparison_path, show=False)
            print(f"  Comparison plot saved to: {comparison_path}")
            
            # Plot convergence for metaheuristics
            meta_stats = [r['stats'] for r in results.values() 
                         if 'convergence_history' in r['stats']]
            if meta_stats:
                convergence_path = os.path.join(args.output_dir, 'convergence.png')
                plot_convergence(meta_stats, save_path=convergence_path, show=False)
                print(f"  Convergence plot saved to: {convergence_path}")
    
    else:
        # Run single algorithm
        print(f"\nRunning {args.algorithm}...")
        
        if args.algorithm == 'nn':
            solution, cost, stats = nearest_neighbor_vrptw(vrptw_instance, args.n_vehicles)
        elif args.algorithm == 'parallel-nn':
            solution, cost, stats = parallel_nearest_neighbor_vrptw(vrptw_instance, args.n_vehicles)
        elif args.algorithm == 'savings':
            solution, cost, stats = clarke_wright_savings_vrptw(vrptw_instance, args.n_vehicles)
        elif args.algorithm == 'parallel-savings':
            solution, cost, stats = parallel_savings_vrptw(vrptw_instance, args.n_vehicles)
        elif args.algorithm == 'ga':
            ga = GeneticAlgorithmVRPTW(vrptw_instance, population_size=50, n_generations=100)
            solution, cost, stats = ga.solve()
        elif args.algorithm == 'sa':
            sa = SimulatedAnnealingVRPTW(vrptw_instance)
            solution, cost, stats = sa.solve()
        
        print_solution_summary(vrptw_instance, solution, stats)
        
        # Save solution
        output_file = os.path.join(args.output_dir, f'{args.algorithm}_solution.txt')
        vrptw_instance.save_solution(solution, output_file)
        print(f"Solution saved to: {output_file}")
        
        # Plot
        if not args.no_plot:
            plot_path = os.path.join(args.output_dir, f'{args.algorithm}_solution.png')
            plot_vrptw_solution(vrptw_instance, solution, 
                              title=f"{stats['algorithm']}", 
                              save_path=plot_path, show=False)
            print(f"Solution plot saved to: {plot_path}")
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETED")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

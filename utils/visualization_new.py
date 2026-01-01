"""
Visualization Utilities for Capacity-Free VRPTW
Professional plotting for routes, algorithm comparison, and analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os
from problems.vrptw import VRPTWInstance


def plot_vrptw_solution(vrptw_instance: VRPTWInstance, 
                        solution: List[List[int]], 
                        title: str = "VRPTW Solution",
                        save_path: str = None) -> None:
    """
    Plot VRPTW solution with routes and time windows
    
    Args:
        vrptw_instance: VRPTW problem instance
        solution: solution as list of routes
        title: plot title
        save_path: path to save plot (optional)
    """
    plt.figure(figsize=(12, 10))
    
    # Colors for different routes
    colors = plt.cm.Set1(np.linspace(0, 1, len(solution)))
    
    # Plot depot
    plt.scatter(vrptw_instance.depot.x, vrptw_instance.depot.y, 
               c='red', s=200, marker='s', label='Depot', zorder=5)
    plt.annotate('DEPOT', (vrptw_instance.depot.x, vrptw_instance.depot.y), 
                xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    # Plot customers with time window info
    for customer in vrptw_instance.customers:
        plt.scatter(customer.x, customer.y, c='lightblue', s=100, 
                  edgecolors='black', alpha=0.7, zorder=3)
        plt.annotate(f'C{customer.id}', (customer.x, customer.y), 
                   xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    # Plot routes
    total_distance = 0
    for i, route in enumerate(solution):
        if not route:
            continue
            
        color = colors[i]
        
        # Route from depot
        current_x, current_y = vrptw_instance.depot.x, vrptw_instance.depot.y
        
        # Plot route segments
        for customer_id in route:
            customer = vrptw_instance.nodes[customer_id]
            next_x, next_y = customer.x, customer.y
            
            # Draw arrow from current to next
            plt.annotate('', xy=(next_x, next_y), xytext=(current_x, current_y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8))
            
            current_x, current_y = next_x, next_y
        
        # Return to depot
        plt.annotate('', xy=(vrptw_instance.depot.x, vrptw_instance.depot.y),
                   xytext=(current_x, current_y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8))
        
        # Calculate route distance
        route_distance = vrptw_instance.calculate_route_distance(route)
        total_distance += route_distance
        
        # Add route label
        if route:
            mid_customer = vrptw_instance.nodes[route[len(route)//2]]
            plt.text(mid_customer.x, mid_customer.y + 2, f'R{i+1}\n{route_distance:.1f}',
                    ha='center', va='bottom', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    # Formatting
    plt.title(f'{title}\nTotal Distance: {total_distance:.2f}, Routes: {len(solution)}', 
             fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Solution plot saved to {save_path}")
    
    plt.show()


def plot_algorithm_comparison(results: Dict[str, Dict], 
                         save_path: str = None) -> None:
    """
    Plot algorithm comparison with multiple metrics
    
    Args:
        results: dictionary of algorithm results
        save_path: path to save plot (optional)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    algorithms = list(results.keys())
    
    # Extract metrics
    costs = [results[alg]['total_cost'] for alg in algorithms]
    routes = [results[alg]['routes'] for alg in algorithms]
    times = [results[alg]['time'] for alg in algorithms]
    feasible = [results[alg]['feasible'] for alg in algorithms]
    
    # Color scheme
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    
    # 1. Total Cost Comparison
    bars1 = ax1.bar(algorithms, costs, color=colors, alpha=0.8)
    ax1.set_title('Total Distance Comparison', fontweight='bold')
    ax1.set_ylabel('Total Distance')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cost in zip(bars1, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{cost:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Number of Routes Comparison
    bars2 = ax2.bar(algorithms, routes, color=colors, alpha=0.8)
    ax2.set_title('Number of Routes Comparison', fontweight='bold')
    ax2.set_ylabel('Number of Routes')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    for bar, route_count in zip(bars2, routes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{route_count}', ha='center', va='bottom', fontsize=9)
    
    # 3. Computation Time Comparison
    bars3 = ax3.bar(algorithms, times, color=colors, alpha=0.8)
    ax3.set_title('Computation Time Comparison', fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for bar, time_val in zip(bars3, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{time_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Feasibility Status
    feasible_colors = ['green' if f else 'red' for f in feasible]
    bars4 = ax4.bar(algorithms, [1]*len(algorithms), color=feasible_colors, alpha=0.8)
    ax4.set_title('Feasibility Status', fontweight='bold')
    ax4.set_ylabel('Status')
    ax4.set_ylim(0, 1.2)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add feasibility labels
    for bar, f in zip(bars4, feasible):
        height = bar.get_height()
        label = '✓' if f else '✗'
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                 label, ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Comparison plot saved to {save_path}")
    
    plt.show()


def plot_pipeline_analysis(pipeline_results: Dict, save_path: str = None) -> None:
    """
    Plot detailed dual-pipeline analysis
    
    Args:
        pipeline_results: results from dual-pipeline framework
        save_path: path to save plot (optional)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract pipeline data
    pipeline_a = pipeline_results['pipeline_a']
    pipeline_b = pipeline_results['pipeline_b']
    
    # 1. Cost progression through stages
    stages = ['Construction', 'After 2-opt', 'After Relocation']
    costs_a = [
        pipeline_a['stages']['construction']['cost'],
        pipeline_a['stages']['construction']['cost'] - pipeline_a['stages']['local_search']['two_opt_improvement'],
        pipeline_a['total_cost']
    ]
    costs_b = [
        pipeline_b['stages']['construction']['cost'],
        pipeline_b['stages']['construction']['cost'] - pipeline_b['stages']['local_search']['two_opt_improvement'],
        pipeline_b['total_cost']
    ]
    
    ax1.plot(stages, costs_a, 'o-', linewidth=2, markersize=8, label='Pipeline A (NN)', color='blue')
    ax1.plot(stages, costs_b, 's-', linewidth=2, markersize=8, label='Pipeline B (CW)', color='red')
    ax1.set_title('Cost Improvement Through Stages', fontweight='bold')
    ax1.set_ylabel('Total Distance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement amounts
    improvements_a = [
        pipeline_a['stages']['local_search']['two_opt_improvement'],
        pipeline_a['stages']['local_search']['relocation_improvement']
    ]
    improvements_b = [
        pipeline_b['stages']['local_search']['two_opt_improvement'],
        pipeline_b['stages']['local_search']['relocation_improvement']
    ]
    
    x = np.arange(2)
    width = 0.35
    
    ax2.bar(x - width/2, improvements_a, width, label='Pipeline A (NN)', alpha=0.8, color='blue')
    ax2.bar(x + width/2, improvements_b, width, label='Pipeline B (CW)', alpha=0.8, color='red')
    ax2.set_title('Local Search Improvements', fontweight='bold')
    ax2.set_ylabel('Distance Reduction')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['2-opt', 'Relocation'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Route count comparison
    routes_data = {
        'Construction A': pipeline_a['stages']['construction']['routes'],
        'Construction B': pipeline_b['stages']['construction']['routes'],
        'Final A': pipeline_a['final_routes'],
        'Final B': pipeline_b['final_routes']
    }
    
    colors_routes = ['lightblue', 'lightcoral', 'blue', 'red']
    bars3 = ax3.bar(list(routes_data.keys()), list(routes_data.values()), color=colors_routes, alpha=0.8)
    ax3.set_title('Route Count Comparison', fontweight='bold')
    ax3.set_ylabel('Number of Routes')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, routes_data.values()):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{value}', ha='center', va='bottom', fontsize=9)
    
    # 4. Time analysis
    time_data = {
        'Construction A': pipeline_a['stages']['construction']['time'],
        'Local Search A': pipeline_a['stages']['local_search']['time'],
        'Construction B': pipeline_b['stages']['construction']['time'],
        'Local Search B': pipeline_b['stages']['local_search']['time']
    }
    
    colors_time = ['lightblue', 'blue', 'lightcoral', 'red']
    bars4 = ax4.bar(list(time_data.keys()), list(time_data.values()), color=colors_time, alpha=0.8)
    ax4.set_title('Computation Time Analysis', fontweight='bold')
    ax4.set_ylabel('Time (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, time_data.values()):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Dual-Pipeline Analysis: {pipeline_results["comparison"]["winner"]} Pipeline Wins', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Pipeline analysis plot saved to {save_path}")
    
    plt.show()


def print_solution_summary(vrptw_instance: VRPTWInstance, 
                        solution: List[List[int]], 
                        cost: float, 
                        algorithm_name: str = "Algorithm") -> None:
    """
    Print detailed solution summary
    
    Args:
        vrptw_instance: VRPTW problem instance
        solution: solution routes
        cost: total cost
        algorithm_name: name of algorithm
    """
    print(f"\n{'='*60}")
    print(f"SOLUTION SUMMARY: {algorithm_name}")
    print(f"{'='*60}")
    print(f"Instance: {vrptw_instance.name}")
    print(f"Customers: {vrptw_instance.n_customers}")
    print(f"Total Distance: {cost:.2f}")
    print(f"Number of Routes: {len(solution)}")
    
    feasible, reason = vrptw_instance.is_solution_feasible(solution)
    print(f"Feasible: {'Yes' if feasible else 'No'}")
    if not feasible:
        print(f"Reason: {reason}")
    
    print(f"\nRoute Details:")
    for i, route in enumerate(solution, 1):
        if route:
            route_cost = vrptw_instance.calculate_route_distance(route)
            print(f"  Route {i}: 0 -> {' -> '.join(map(str, route))} -> 0")
            print(f"    Distance: {route_cost:.2f}, Customers: {len(route)}")
            
            # Time window analysis
            current_time = vrptw_instance.depot.ready_time
            current_node = 0
            
            for customer_id in route:
                travel_time = vrptw_instance.get_travel_time(current_node, customer_id)
                arrival_time = current_time + travel_time
                service_start = max(arrival_time, vrptw_instance.nodes[customer_id].ready_time)
                waiting_time = max(0, service_start - arrival_time)
                
                print(f"      C{customer_id}: arrive={arrival_time:.1f}, "
                      f"service={service_start:.1f}, wait={waiting_time:.1f}")
                
                current_time = service_start + vrptw_instance.nodes[customer_id].service_time
                current_node = customer_id
    
    print(f"{'='*60}")


def save_results_to_csv(results: Dict, filename: str) -> None:
    """
    Save algorithm results to CSV file
    
    Args:
        results: dictionary of algorithm results
        filename: output CSV filename
    """
    data = []
    for algorithm, stats in results.items():
        row = {
            'Algorithm': algorithm,
            'Total_Cost': stats['total_cost'],
            'Routes': stats['routes'],
            'Time': stats['time'],
            'Feasible': stats['feasible']
        }
        
        # Add pipeline-specific fields if available
        if 'stages' in stats:
            row.update({
                'Construction_Cost': stats['stages']['construction']['cost'],
                'Construction_Routes': stats['stages']['construction']['routes'],
                'Construction_Time': stats['stages']['construction']['time'],
                'Total_Improvement': stats['total_improvement']
            })
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"  Results saved to {filename}")

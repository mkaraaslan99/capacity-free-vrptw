"""
2-Opt Local Search Heuristic for TSP
Iteratively improves a tour by removing two edges and reconnecting the tour in a different way
"""
import numpy as np
from typing import List, Tuple
import time


def calculate_2opt_delta(tsp_instance, tour: List[int], i: int, j: int) -> float:
    """
    Calculate the change in tour length from a 2-opt move without creating new tour
    
    Args:
        tsp_instance: TSPInstance object
        tour: current tour
        i, j: indices for 2-opt swap
        
    Returns:
        change in tour length (negative means improvement)
    """
    n = len(tour)
    
    # Current edges
    a, b = tour[i], tour[(i + 1) % n]
    c, d = tour[j], tour[(j + 1) % n]
    
    # Current distance
    current_dist = tsp_instance.get_distance(a, b) + tsp_instance.get_distance(c, d)
    
    # New distance after swap
    new_dist = tsp_instance.get_distance(a, c) + tsp_instance.get_distance(b, d)
    
    return new_dist - current_dist


def two_opt_swap(tour: List[int], i: int, j: int) -> List[int]:
    """
    Perform 2-opt swap on a tour
    
    Args:
        tour: original tour
        i, j: indices to swap (reverses segment between i+1 and j)
        
    Returns:
        new tour after 2-opt swap
    """
    new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
    return new_tour


def two_opt_tsp(tsp_instance, initial_tour: List[int] = None, max_iterations: int = 1000) -> Tuple[List[int], float, dict]:
    """
    Improve TSP tour using 2-opt local search
    
    Args:
        tsp_instance: TSPInstance object
        initial_tour: starting tour (if None, generates random tour)
        max_iterations: maximum number of iterations without improvement
        
    Returns:
        tuple of (tour, tour_length, stats)
    """
    start_time = time.time()
    
    if initial_tour is None:
        tour = tsp_instance.generate_random_tour()
    else:
        tour = initial_tour.copy()
    
    n = len(tour)
    improved = True
    iterations = 0
    improvements = 0
    
    best_length = tsp_instance.calculate_tour_length(tour)
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip adjacent edges
                if j == i + 1:
                    continue
                
                # Calculate improvement
                delta = calculate_2opt_delta(tsp_instance, tour, i, j)
                
                if delta < -1e-10:  # Improvement found
                    tour = two_opt_swap(tour, i, j)
                    best_length += delta
                    improved = True
                    improvements += 1
                    break
            
            if improved:
                break
    
    final_length = tsp_instance.calculate_tour_length(tour)
    elapsed_time = time.time() - start_time
    
    stats = {
        'algorithm': '2-Opt',
        'tour_length': final_length,
        'time': elapsed_time,
        'iterations': iterations,
        'improvements': improvements
    }
    
    return tour, final_length, stats


def two_opt_first_improvement(tsp_instance, initial_tour: List[int] = None, max_iterations: int = 1000) -> Tuple[List[int], float, dict]:
    """
    2-Opt with first improvement strategy (faster but potentially lower quality)
    
    Args:
        tsp_instance: TSPInstance object
        initial_tour: starting tour
        max_iterations: maximum iterations without improvement
        
    Returns:
        tuple of (tour, tour_length, stats)
    """
    start_time = time.time()
    
    if initial_tour is None:
        tour = tsp_instance.generate_random_tour()
    else:
        tour = initial_tour.copy()
    
    n = len(tour)
    improved = True
    iterations = 0
    improvements = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == i + 1:
                    continue
                
                delta = calculate_2opt_delta(tsp_instance, tour, i, j)
                
                if delta < -1e-10:
                    tour = two_opt_swap(tour, i, j)
                    improved = True
                    improvements += 1
                    break
            
            if improved:
                break
    
    final_length = tsp_instance.calculate_tour_length(tour)
    elapsed_time = time.time() - start_time
    
    stats = {
        'algorithm': '2-Opt (First Improvement)',
        'tour_length': final_length,
        'time': elapsed_time,
        'iterations': iterations,
        'improvements': improvements
    }
    
    return tour, final_length, stats

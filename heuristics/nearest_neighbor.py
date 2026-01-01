"""
Nearest Neighbor Heuristic for TSP
A greedy constructive heuristic that builds a tour by always visiting the nearest unvisited city
"""
import numpy as np
from typing import List, Tuple
import time


def nearest_neighbor_tsp(tsp_instance, start_city: int = 0) -> Tuple[List[int], float, dict]:
    """
    Solve TSP using Nearest Neighbor heuristic
    
    Args:
        tsp_instance: TSPInstance object
        start_city: starting city index
        
    Returns:
        tuple of (tour, tour_length, stats)
    """
    start_time = time.time()
    
    n = tsp_instance.n_cities
    unvisited = set(range(n))
    tour = [start_city]
    unvisited.remove(start_city)
    
    current_city = start_city
    
    # Build tour by always going to nearest unvisited city
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: tsp_instance.get_distance(current_city, city))
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
    
    tour_length = tsp_instance.calculate_tour_length(tour)
    elapsed_time = time.time() - start_time
    
    stats = {
        'algorithm': 'Nearest Neighbor',
        'tour_length': tour_length,
        'time': elapsed_time,
        'start_city': start_city
    }
    
    return tour, tour_length, stats


def multi_start_nearest_neighbor(tsp_instance, n_starts: int = None) -> Tuple[List[int], float, dict]:
    """
    Run Nearest Neighbor from multiple starting cities and return best solution
    
    Args:
        tsp_instance: TSPInstance object
        n_starts: number of different starting cities to try (default: all cities)
        
    Returns:
        tuple of (best_tour, best_length, stats)
    """
    start_time = time.time()
    
    if n_starts is None:
        n_starts = tsp_instance.n_cities
    
    best_tour = None
    best_length = float('inf')
    
    for start_city in range(min(n_starts, tsp_instance.n_cities)):
        tour, length, _ = nearest_neighbor_tsp(tsp_instance, start_city)
        
        if length < best_length:
            best_length = length
            best_tour = tour
    
    elapsed_time = time.time() - start_time
    
    stats = {
        'algorithm': 'Multi-Start Nearest Neighbor',
        'tour_length': best_length,
        'time': elapsed_time,
        'n_starts': n_starts
    }
    
    return best_tour, best_length, stats

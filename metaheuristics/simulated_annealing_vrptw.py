"""
Simulated Annealing for VRPTW
Probabilistic metaheuristic inspired by metallurgical annealing
"""
import numpy as np
import random
from typing import List, Tuple, Dict
import time
import copy
import math


class SimulatedAnnealingVRPTW:
    """Simulated Annealing solver for VRPTW"""
    
    def __init__(self, vrptw_instance, initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995, min_temp: float = 0.1,
                 iterations_per_temp: int = 100):
        """
        Initialize Simulated Annealing
        
        Args:
            vrptw_instance: VRPTWInstance object
            initial_temp: starting temperature
            cooling_rate: temperature reduction factor (0 < rate < 1)
            min_temp: stopping temperature
            iterations_per_temp: iterations at each temperature
        """
        self.instance = vrptw_instance
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp
    
    def objective(self, solution: List[List[int]]) -> float:
        """Calculate objective value with penalty for infeasibility"""
        cost = self.instance.calculate_solution_cost(solution)
        feasible, _ = self.instance.is_solution_feasible(solution)
        
        if not feasible:
            cost *= 5  # Penalty for infeasibility
        
        return cost
    
    def get_neighbor_swap_customers(self, solution: List[List[int]]) -> List[List[int]]:
        """Generate neighbor by swapping two customers"""
        new_solution = copy.deepcopy(solution)
        
        # Get all non-empty routes
        non_empty_routes = [i for i, route in enumerate(new_solution) if route]
        
        if len(non_empty_routes) < 1:
            return new_solution
        
        # Swap within same route or between routes
        if len(non_empty_routes) >= 2 and random.random() < 0.5:
            # Swap between routes
            route1_idx, route2_idx = random.sample(non_empty_routes, 2)
            route1 = new_solution[route1_idx]
            route2 = new_solution[route2_idx]
            
            if route1 and route2:
                pos1 = random.randint(0, len(route1) - 1)
                pos2 = random.randint(0, len(route2) - 1)
                route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
        else:
            # Swap within route
            route_idx = random.choice(non_empty_routes)
            route = new_solution[route_idx]
            
            if len(route) >= 2:
                pos1, pos2 = random.sample(range(len(route)), 2)
                route[pos1], route[pos2] = route[pos2], route[pos1]
        
        return new_solution
    
    def get_neighbor_relocate(self, solution: List[List[int]]) -> List[List[int]]:
        """Generate neighbor by relocating a customer to different position"""
        new_solution = copy.deepcopy(solution)
        
        non_empty_routes = [i for i, route in enumerate(new_solution) if route]
        
        if not non_empty_routes:
            return new_solution
        
        # Remove customer from one route
        from_route_idx = random.choice(non_empty_routes)
        from_route = new_solution[from_route_idx]
        
        if not from_route:
            return new_solution
        
        customer_pos = random.randint(0, len(from_route) - 1)
        customer = from_route.pop(customer_pos)
        
        # Insert into another route (or same route at different position)
        if len(new_solution) > 1:
            to_route_idx = random.randint(0, len(new_solution) - 1)
        else:
            to_route_idx = 0
        
        to_route = new_solution[to_route_idx]
        insert_pos = random.randint(0, len(to_route))
        to_route.insert(insert_pos, customer)
        
        # Remove empty routes
        new_solution = [route for route in new_solution if route]
        
        return new_solution
    
    def get_neighbor_2opt(self, solution: List[List[int]]) -> List[List[int]]:
        """Generate neighbor using 2-opt within a route"""
        new_solution = copy.deepcopy(solution)
        
        non_empty_routes = [i for i, route in enumerate(new_solution) if len(route) >= 2]
        
        if not non_empty_routes:
            return new_solution
        
        route_idx = random.choice(non_empty_routes)
        route = new_solution[route_idx]
        
        if len(route) >= 2:
            i, j = sorted(random.sample(range(len(route)), 2))
            route[i:j+1] = route[i:j+1][::-1]
        
        return new_solution
    
    def get_neighbor(self, solution: List[List[int]]) -> List[List[int]]:
        """Generate a random neighbor solution"""
        operation = random.choice(['swap', 'relocate', '2opt'])
        
        if operation == 'swap':
            return self.get_neighbor_swap_customers(solution)
        elif operation == 'relocate':
            return self.get_neighbor_relocate(solution)
        else:
            return self.get_neighbor_2opt(solution)
    
    def acceptance_probability(self, current_cost: float, new_cost: float, temperature: float) -> float:
        """Calculate probability of accepting worse solution"""
        if new_cost < current_cost:
            return 1.0
        return math.exp(-(new_cost - current_cost) / temperature)
    
    def solve(self, initial_solution: List[List[int]] = None) -> Tuple[List[List[int]], float, Dict]:
        """
        Run simulated annealing
        
        Args:
            initial_solution: starting solution (if None, creates random solution)
            
        Returns:
            tuple of (best_solution, best_cost, stats)
        """
        start_time = time.time()
        
        # Initialize
        if initial_solution is None:
            # Create simple initial solution
            customers = list(range(1, self.instance.n_nodes))
            random.shuffle(customers)
            current_solution = [[c] for c in customers]
        else:
            current_solution = copy.deepcopy(initial_solution)
        
        current_cost = self.objective(current_solution)
        
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        temperature = self.initial_temp
        convergence_history = []
        acceptances = 0
        rejections = 0
        iterations = 0
        
        while temperature > self.min_temp:
            for _ in range(self.iterations_per_temp):
                iterations += 1
                
                # Generate neighbor
                new_solution = self.get_neighbor(current_solution)
                new_cost = self.objective(new_solution)
                
                # Acceptance criterion
                if random.random() < self.acceptance_probability(current_cost, new_cost, temperature):
                    current_solution = new_solution
                    current_cost = new_cost
                    acceptances += 1
                    
                    # Update best
                    if current_cost < best_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_cost = current_cost
                else:
                    rejections += 1
            
            convergence_history.append(best_cost)
            temperature *= self.cooling_rate
        
        elapsed_time = time.time() - start_time
        
        # Calculate actual cost without penalty
        actual_cost = self.instance.calculate_solution_cost(best_solution)
        feasible, reason = self.instance.is_solution_feasible(best_solution)
        
        stats = {
            'algorithm': 'Simulated Annealing VRPTW',
            'total_cost': actual_cost,
            'n_routes': len(best_solution),
            'time': elapsed_time,
            'iterations': iterations,
            'acceptances': acceptances,
            'rejections': rejections,
            'acceptance_rate': acceptances / (acceptances + rejections) if (acceptances + rejections) > 0 else 0,
            'convergence_history': convergence_history,
            'feasible': feasible,
            'feasibility_reason': reason
        }
        
        return best_solution, actual_cost, stats

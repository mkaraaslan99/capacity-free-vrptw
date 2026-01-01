"""
Genetic Algorithm for VRPTW
Population-based evolutionary approach with specialized operators for routing problems
"""
import numpy as np
import random
from typing import List, Tuple, Dict
import time
import copy


class GeneticAlgorithmVRPTW:
    """Genetic Algorithm solver for VRPTW"""
    
    def __init__(self, vrptw_instance, population_size: int = 50, 
                 n_generations: int = 100, mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8, elite_size: int = 5):
        """
        Initialize GA
        
        Args:
            vrptw_instance: VRPTWInstance object
            population_size: number of solutions in population
            n_generations: number of generations to evolve
            mutation_rate: probability of mutation
            crossover_rate: probability of crossover
            elite_size: number of best solutions to preserve
        """
        self.instance = vrptw_instance
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
    
    def create_random_solution(self) -> List[List[int]]:
        """Create a random feasible solution"""
        customers = list(range(1, self.instance.n_nodes))
        random.shuffle(customers)
        
        # Split into routes
        solution = []
        current_route = []
        current_time = self.instance.depot.ready_time
        current_load = 0
        current_node = 0
        
        for customer_id in customers:
            customer = self.instance.nodes[customer_id]
            
            # Check if we can add to current route
            travel_time = self.instance.get_distance(current_node, customer_id)
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, customer.ready_time)
            finish_time = service_start + customer.service_time
            return_time = finish_time + self.instance.get_distance(customer_id, 0)
            
            can_add = (current_load + customer.demand <= self.instance.vehicle_capacity and
                      service_start <= customer.due_time and
                      return_time <= self.instance.depot.due_time)
            
            if can_add and current_route:
                current_route.append(customer_id)
                current_time = finish_time
                current_load += customer.demand
                current_node = customer_id
            else:
                # Start new route
                if current_route:
                    solution.append(current_route)
                current_route = [customer_id]
                current_time = self.instance.depot.ready_time + self.instance.get_distance(0, customer_id)
                current_time = max(current_time, customer.ready_time) + customer.service_time
                current_load = customer.demand
                current_node = customer_id
        
        if current_route:
            solution.append(current_route)
        
        return solution
    
    def initialize_population(self) -> List[List[List[int]]]:
        """Create initial population"""
        population = []
        for _ in range(self.population_size):
            solution = self.create_random_solution()
            population.append(solution)
        return population
    
    def fitness(self, solution: List[List[int]]) -> float:
        """
        Calculate fitness (lower is better)
        Penalize infeasible solutions
        """
        cost = self.instance.calculate_solution_cost(solution)
        feasible, _ = self.instance.is_solution_feasible(solution)
        
        if not feasible:
            cost *= 10  # Heavy penalty for infeasibility
        
        return cost
    
    def tournament_selection(self, population: List, fitness_scores: List[float], 
                           tournament_size: int = 3) -> List[List[int]]:
        """Select parent using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return copy.deepcopy(population[winner_idx])
    
    def order_crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """
        Order crossover for VRPTW
        Preserves relative order of customers from parents
        """
        # Flatten routes to customer sequences
        seq1 = [c for route in parent1 for c in route]
        seq2 = [c for route in parent2 for c in route]
        
        # Perform order crossover on sequences
        size = len(seq1)
        start, end = sorted(random.sample(range(size), 2))
        
        child_seq = [-1] * size
        child_seq[start:end] = seq1[start:end]
        
        # Fill remaining positions from parent2
        pos = end
        for customer in seq2:
            if customer not in child_seq:
                if pos >= size:
                    pos = 0
                child_seq[pos] = customer
                pos += 1
        
        # Split back into routes
        return self.split_into_routes(child_seq)
    
    def split_into_routes(self, customer_sequence: List[int]) -> List[List[int]]:
        """Split customer sequence into feasible routes"""
        solution = []
        current_route = []
        current_time = self.instance.depot.ready_time
        current_load = 0
        current_node = 0
        
        for customer_id in customer_sequence:
            customer = self.instance.nodes[customer_id]
            
            travel_time = self.instance.get_distance(current_node, customer_id)
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, customer.ready_time)
            finish_time = service_start + customer.service_time
            return_time = finish_time + self.instance.get_distance(customer_id, 0)
            
            can_add = (current_load + customer.demand <= self.instance.vehicle_capacity and
                      service_start <= customer.due_time and
                      return_time <= self.instance.depot.due_time)
            
            if can_add and current_route:
                current_route.append(customer_id)
                current_time = finish_time
                current_load += customer.demand
                current_node = customer_id
            else:
                if current_route:
                    solution.append(current_route)
                current_route = [customer_id]
                current_time = self.instance.depot.ready_time + self.instance.get_distance(0, customer_id)
                current_time = max(current_time, customer.ready_time) + customer.service_time
                current_load = customer.demand
                current_node = customer_id
        
        if current_route:
            solution.append(current_route)
        
        return solution
    
    def mutate_swap(self, solution: List[List[int]]) -> List[List[int]]:
        """Swap mutation - swap two random customers"""
        solution = copy.deepcopy(solution)
        
        # Flatten to get all customers
        all_customers = [c for route in solution for c in route]
        
        if len(all_customers) < 2:
            return solution
        
        # Swap two random customers
        idx1, idx2 = random.sample(range(len(all_customers)), 2)
        all_customers[idx1], all_customers[idx2] = all_customers[idx2], all_customers[idx1]
        
        # Rebuild routes
        return self.split_into_routes(all_customers)
    
    def mutate_inversion(self, solution: List[List[int]]) -> List[List[int]]:
        """Inversion mutation - reverse a random segment"""
        solution = copy.deepcopy(solution)
        
        if not solution or not any(solution):
            return solution
        
        # Choose random route
        route_idx = random.randint(0, len(solution) - 1)
        route = solution[route_idx]
        
        if len(route) < 2:
            return solution
        
        # Reverse a segment
        start, end = sorted(random.sample(range(len(route)), 2))
        route[start:end+1] = route[start:end+1][::-1]
        
        return solution
    
    def solve(self) -> Tuple[List[List[int]], float, Dict]:
        """
        Run genetic algorithm
        
        Returns:
            tuple of (best_solution, best_cost, stats)
        """
        start_time = time.time()
        
        # Initialize
        population = self.initialize_population()
        
        best_solution = None
        best_cost = float('inf')
        best_generation = 0
        
        convergence_history = []
        
        for generation in range(self.n_generations):
            # Evaluate fitness
            fitness_scores = [self.fitness(sol) for sol in population]
            
            # Track best solution
            min_fitness_idx = fitness_scores.index(min(fitness_scores))
            if fitness_scores[min_fitness_idx] < best_cost:
                best_cost = fitness_scores[min_fitness_idx]
                best_solution = copy.deepcopy(population[min_fitness_idx])
                best_generation = generation
            
            convergence_history.append(best_cost)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best solutions
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:self.elite_size]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(population[idx]))
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    if random.random() < 0.5:
                        child = self.mutate_swap(child)
                    else:
                        child = self.mutate_inversion(child)
                
                new_population.append(child)
            
            population = new_population
        
        elapsed_time = time.time() - start_time
        
        # Recalculate actual cost (without penalty)
        actual_cost = self.instance.calculate_solution_cost(best_solution)
        feasible, reason = self.instance.is_solution_feasible(best_solution)
        
        stats = {
            'algorithm': 'Genetic Algorithm VRPTW',
            'total_cost': actual_cost,
            'n_routes': len(best_solution),
            'time': elapsed_time,
            'best_generation': best_generation,
            'convergence_history': convergence_history,
            'feasible': feasible,
            'feasibility_reason': reason
        }
        
        return best_solution, actual_cost, stats

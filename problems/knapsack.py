"""
Knapsack Problem Definition and Utilities
"""
import numpy as np
from typing import List, Tuple
import random


class KnapsackInstance:
    """Represents a Knapsack problem instance"""
    
    def __init__(self, weights: List[float], values: List[float], capacity: float, name: str = "Knapsack"):
        """
        Initialize Knapsack instance
        
        Args:
            weights: list of item weights
            values: list of item values
            capacity: knapsack capacity
            name: name of the instance
        """
        assert len(weights) == len(values), "Weights and values must have same length"
        
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.n_items = len(weights)
        self.name = name
    
    def evaluate_solution(self, solution: List[int]) -> Tuple[float, bool]:
        """
        Evaluate a solution (binary vector indicating selected items)
        
        Args:
            solution: binary list where 1 means item is selected
            
        Returns:
            tuple of (total_value, is_feasible)
        """
        total_weight = np.sum(self.weights * solution)
        total_value = np.sum(self.values * solution)
        is_feasible = total_weight <= self.capacity
        
        return total_value, is_feasible
    
    def get_weight(self, solution: List[int]) -> float:
        """Get total weight of a solution"""
        return np.sum(self.weights * solution)
    
    def get_value(self, solution: List[int]) -> float:
        """Get total value of a solution"""
        return np.sum(self.values * solution)
    
    def is_feasible(self, solution: List[int]) -> bool:
        """Check if solution is feasible (doesn't exceed capacity)"""
        return self.get_weight(solution) <= self.capacity
    
    def generate_random_solution(self) -> List[int]:
        """Generate a random feasible solution"""
        solution = [0] * self.n_items
        indices = list(range(self.n_items))
        random.shuffle(indices)
        
        for idx in indices:
            if self.weights[idx] + self.get_weight(solution) <= self.capacity:
                solution[idx] = 1
        
        return solution
    
    @staticmethod
    def generate_random_instance(n_items: int, capacity_ratio: float = 0.5) -> 'KnapsackInstance':
        """
        Generate a random Knapsack instance
        
        Args:
            n_items: number of items
            capacity_ratio: capacity as ratio of total weight
            
        Returns:
            KnapsackInstance object
        """
        weights = np.random.randint(1, 100, n_items).tolist()
        values = np.random.randint(1, 100, n_items).tolist()
        capacity = sum(weights) * capacity_ratio
        
        return KnapsackInstance(weights, values, capacity, name=f"Random_{n_items}")
    
    @staticmethod
    def load_from_file(filename: str) -> 'KnapsackInstance':
        """
        Load Knapsack instance from file
        
        File format:
        n_items capacity
        value1 weight1
        value2 weight2
        ...
        """
        with open(filename, 'r') as f:
            n_items, capacity = map(float, f.readline().split())
            n_items = int(n_items)
            
            values = []
            weights = []
            
            for _ in range(n_items):
                v, w = map(float, f.readline().split())
                values.append(v)
                weights.append(w)
        
        name = filename.split('/')[-1].replace('.txt', '')
        return KnapsackInstance(weights, values, capacity, name=name)
    
    def save_to_file(self, filename: str):
        """Save Knapsack instance to file"""
        with open(filename, 'w') as f:
            f.write(f"{self.n_items} {self.capacity}\n")
            for v, w in zip(self.values, self.weights):
                f.write(f"{v} {w}\n")


def validate_solution(solution: List[int], n_items: int) -> bool:
    """
    Validate if a solution is valid (binary vector of correct length)
    
    Args:
        solution: binary list
        n_items: expected number of items
        
    Returns:
        True if valid, False otherwise
    """
    if len(solution) != n_items:
        return False
    if not all(x in [0, 1] for x in solution):
        return False
    return True

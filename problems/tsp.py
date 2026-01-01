"""
Traveling Salesman Problem (TSP) Definition and Utilities
"""
import numpy as np
import random
from typing import List, Tuple


class TSPInstance:
    """Represents a TSP problem instance"""
    
    def __init__(self, cities: np.ndarray, name: str = "TSP"):
        """
        Initialize TSP instance
        
        Args:
            cities: numpy array of shape (n, 2) with city coordinates
            name: name of the instance
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.name = name
        self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute Euclidean distance matrix between all cities"""
        n = self.n_cities
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.cities[i] - self.cities[j])
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        
        return dist_matrix
    
    def calculate_tour_length(self, tour: List[int]) -> float:
        """
        Calculate total length of a tour
        
        Args:
            tour: list of city indices representing the tour
            
        Returns:
            total tour length
        """
        total_length = 0.0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            total_length += self.distance_matrix[from_city][to_city]
        return total_length
    
    def get_distance(self, city1: int, city2: int) -> float:
        """Get distance between two cities"""
        return self.distance_matrix[city1][city2]
    
    def generate_random_tour(self) -> List[int]:
        """Generate a random valid tour"""
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour
    
    @staticmethod
    def generate_random_instance(n_cities: int, width: float = 100.0, height: float = 100.0) -> 'TSPInstance':
        """
        Generate a random TSP instance
        
        Args:
            n_cities: number of cities
            width: width of the area
            height: height of the area
            
        Returns:
            TSPInstance object
        """
        cities = np.random.rand(n_cities, 2) * [width, height]
        return TSPInstance(cities, name=f"Random_{n_cities}")
    
    @staticmethod
    def load_from_file(filename: str) -> 'TSPInstance':
        """
        Load TSP instance from file (TSPLIB format)
        
        Args:
            filename: path to the file
            
        Returns:
            TSPInstance object
        """
        cities = []
        reading_coords = False
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('NODE_COORD_SECTION'):
                    reading_coords = True
                    continue
                if line == 'EOF' or line == '':
                    break
                if reading_coords:
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y = float(parts[1]), float(parts[2])
                        cities.append([x, y])
        
        cities_array = np.array(cities)
        name = filename.split('/')[-1].replace('.tsp', '')
        return TSPInstance(cities_array, name=name)
    
    def save_to_file(self, filename: str):
        """Save TSP instance to file in TSPLIB format"""
        with open(filename, 'w') as f:
            f.write(f"NAME: {self.name}\n")
            f.write(f"TYPE: TSP\n")
            f.write(f"DIMENSION: {self.n_cities}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write(f"NODE_COORD_SECTION\n")
            for i, city in enumerate(self.cities, 1):
                f.write(f"{i} {city[0]:.6f} {city[1]:.6f}\n")
            f.write("EOF\n")


def validate_tour(tour: List[int], n_cities: int) -> bool:
    """
    Validate if a tour is valid (visits all cities exactly once)
    
    Args:
        tour: list of city indices
        n_cities: expected number of cities
        
    Returns:
        True if valid, False otherwise
    """
    if len(tour) != n_cities:
        return False
    if set(tour) != set(range(n_cities)):
        return False
    return True


def two_opt_swap(tour: List[int], i: int, j: int) -> List[int]:
    """
    Perform 2-opt swap on a tour
    
    Args:
        tour: original tour
        i, j: indices to swap
        
    Returns:
        new tour after 2-opt swap
    """
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

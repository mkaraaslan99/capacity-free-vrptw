"""
Solution Encoding for Capacity-Free VRPTW
Multiple encoding schemes for different algorithmic approaches
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from problems.vrptw import VRPTWInstance


class VRPTWSolutionEncoder:
    """
    Handles multiple solution encoding schemes for capacity-free VRPTW
    """
    
    def __init__(self, vrptw_instance: VRPTWInstance):
        self.vrptw_instance = vrptw_instance
        self.n_customers = vrptw_instance.n_customers
    
    def route_based_to_permutation(self, routes: List[List[int]]) -> List[int]:
        """
        Convert route-based encoding to permutation with delimiters
        
        Args:
            routes: List of routes [[c1, c2], [c3, c4, c5]]
            
        Returns:
            Permutation with -1 as route delimiters [c1, c2, -1, c3, c4, c5]
        """
        permutation = []
        for route in routes:
            permutation.extend(route)
            permutation.append(-1)  # Route delimiter
        
        return permutation[:-1]  # Remove last delimiter
    
    def permutation_to_route_based(self, permutation: List[int]) -> List[List[int]]:
        """
        Convert permutation with delimiters back to route-based encoding
        
        Args:
            permutation: List with -1 as delimiters [c1, c2, -1, c3, c4]
            
        Returns:
            Route-based encoding [[c1, c2], [c3, c4]]
        """
        routes = []
        current_route = []
        
        for customer in permutation:
            if customer == -1:
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(customer)
        
        if current_route:
            routes.append(current_route)
        
        return routes
    
    def route_based_to_giant_tour(self, routes: List[List[int]]) -> Tuple[List[int], List[int]]:
        """
        Convert route-based to giant tour with route breaks
        
        Args:
            routes: List of routes
            
        Returns:
            Tuple of (giant_tour, route_breaks)
        """
        giant_tour = []
        route_breaks = []
        customer_count = 0
        
        for route in routes:
            giant_tour.extend(route)
            customer_count += len(route)
            if customer_count < self.n_customers:
                route_breaks.append(customer_count - 1)  # Last customer in this route
        
        return giant_tour, route_breaks
    
    def giant_tour_to_route_based(self, giant_tour: List[int], 
                                 route_breaks: List[int]) -> List[List[int]]:
        """
        Convert giant tour with breaks back to route-based encoding
        
        Args:
            giant_tour: Complete tour of all customers
            route_breaks: Indices where routes end
            
        Returns:
            Route-based encoding
        """
        routes = []
        start_idx = 0
        
        for break_idx in sorted(route_breaks):
            route = giant_tour[start_idx:break_idx + 1]
            routes.append(route)
            start_idx = break_idx + 1
        
        # Add remaining customers
        if start_idx < len(giant_tour):
            routes.append(giant_tour[start_idx:])
        
        return routes
    
    def route_based_to_binary_matrix(self, routes: List[List[int]]) -> np.ndarray:
        """
        Convert route-based to binary adjacency matrix
        
        Args:
            routes: List of routes
            
        Returns:
            n×n binary matrix where matrix[i][j] = 1 if j follows i
        """
        n = self.n_customers + 1  # Include depot
        matrix = np.zeros((n, n), dtype=int)
        
        for route in routes:
            # Depot to first customer
            if route:
                matrix[0][route[0]] = 1
                
                # Customer to customer
                for i in range(len(route) - 1):
                    matrix[route[i]][route[i + 1]] = 1
                
                # Last customer to depot
                matrix[route[-1]][0] = 1
        
        return matrix
    
    def binary_matrix_to_route_based(self, matrix: np.ndarray) -> List[List[int]]:
        """
        Convert binary adjacency matrix back to route-based encoding
        
        Args:
            matrix: n×n binary adjacency matrix
            
        Returns:
            Route-based encoding
        """
        n = self.n_customers + 1
        routes = []
        visited = set()
        
        # Find all routes starting from depot
        for start_customer in range(1, n):
            if start_customer in visited:
                continue
                
            # Check if there's a path from depot to this customer
            if matrix[0][start_customer] == 1:
                route = []
                current = start_customer
                
                while current != 0 and current not in visited:
                    route.append(current)
                    visited.add(current)
                    
                    # Find next customer
                    next_customer = None
                    for j in range(n):
                        if matrix[current][j] == 1 and j not in visited:
                            next_customer = j
                            break
                    
                    if next_customer is None:
                        break
                    current = next_customer
                
                if route:
                    routes.append(route)
        
        return routes
    
    def encode_solution(self, routes: List[List[int]], 
                      encoding_type: str = 'route_based') -> Union[List[List[int]], List[int], Tuple, np.ndarray]:
        """
        Encode solution using specified encoding type
        
        Args:
            routes: Route-based solution
            encoding_type: 'route_based', 'permutation', 'giant_tour', 'binary_matrix'
            
        Returns:
            Encoded solution in specified format
        """
        if encoding_type == 'route_based':
            return routes
        elif encoding_type == 'permutation':
            return self.route_based_to_permutation(routes)
        elif encoding_type == 'giant_tour':
            return self.route_based_to_giant_tour(routes)
        elif encoding_type == 'binary_matrix':
            return self.route_based_to_binary_matrix(routes)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def decode_solution(self, encoded_solution: Union[List[List[int]], List[int], Tuple, np.ndarray],
                      encoding_type: str = 'route_based') -> List[List[int]]:
        """
        Decode solution from specified encoding type
        
        Args:
            encoded_solution: Encoded solution
            encoding_type: 'route_based', 'permutation', 'giant_tour', 'binary_matrix'
            
        Returns:
            Route-based solution
        """
        if encoding_type == 'route_based':
            return encoded_solution
        elif encoding_type == 'permutation':
            return self.permutation_to_route_based(encoded_solution)
        elif encoding_type == 'giant_tour':
            giant_tour, route_breaks = encoded_solution
            return self.giant_tour_to_route_based(giant_tour, route_breaks)
        elif encoding_type == 'binary_matrix':
            return self.binary_matrix_to_route_based(encoded_solution)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def validate_encoding(self, encoded_solution: Union[List[List[int]], List[int], Tuple, np.ndarray],
                       encoding_type: str = 'route_based') -> Tuple[bool, str]:
        """
        Validate encoded solution
        
        Args:
            encoded_solution: Encoded solution
            encoding_type: Type of encoding
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            routes = self.decode_solution(encoded_solution, encoding_type)
            
            # Check all customers are served
            all_customers = set(range(1, self.n_customers + 1))
            served_customers = set()
            
            for route in routes:
                served_customers.update(route)
            
            if served_customers != all_customers:
                missing = all_customers - served_customers
                duplicate = served_customers - all_customers
                return False, f"Missing: {missing}, Duplicate: {duplicate}"
            
            # Check feasibility
            feasible, reason = self.vrptw_instance.is_solution_feasible(routes)
            return feasible, reason
            
        except Exception as e:
            return False, f"Decoding error: {str(e)}"
    
    def generate_random_solution(self, encoding_type: str = 'route_based') -> Union[List[List[int]], List[int], Tuple, np.ndarray]:
        """
        Generate random solution using specified encoding
        
        Args:
            encoding_type: Type of encoding to generate
            
        Returns:
            Random encoded solution
        """
        # Generate random permutation of customers
        customers = list(range(1, self.n_customers + 1))
        np.random.shuffle(customers)
        
        if encoding_type == 'route_based':
            # Random route assignment
            num_routes = np.random.randint(1, min(5, self.n_customers))
            routes = []
            for i in range(num_routes):
                start_idx = i * len(customers) // num_routes
                end_idx = (i + 1) * len(customers) // num_routes if i < num_routes - 1 else len(customers)
                routes.append(customers[start_idx:end_idx])
            return routes
        
        elif encoding_type == 'permutation':
            # Random permutation with random delimiters
            num_delimiters = np.random.randint(0, min(4, self.n_customers - 1))
            delimiter_positions = sorted(np.random.choice(range(len(customers) - 1), num_delimiters, replace=False))
            
            permutation = []
            for i, customer in enumerate(customers):
                permutation.append(customer)
                if i in delimiter_positions:
                    permutation.append(-1)
            return permutation
        
        elif encoding_type == 'giant_tour':
            # Random giant tour with random breaks
            route_breaks = sorted(np.random.choice(range(len(customers) - 1), 
                                              np.random.randint(1, min(4, len(customers) - 1)), 
                                              replace=False))
            return customers, route_breaks
        
        elif encoding_type == 'binary_matrix':
            # Generate random route-based then convert
            routes = self.generate_random_solution('route_based')
            return self.route_based_to_binary_matrix(routes)
        
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")


class SolutionOperators:
    """
    Operators for manipulating encoded solutions
    """
    
    def __init__(self, vrptw_instance: VRPTWInstance):
        self.vrptw_instance = vrptw_instance
        self.encoder = VRPTWSolutionEncoder(vrptw_instance)
    
    def crossover_permutation(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Order crossover (OX) for permutation encoding
        """
        n = len(parent1)
        
        # Remove delimiters for crossover
        p1_clean = [x for x in parent1 if x != -1]
        p2_clean = [x for x in parent2 if x != -1]
        
        # Select crossover points
        start, end = sorted(np.random.choice(range(len(p1_clean)), 2, replace=False))
        
        # Create offspring
        child1 = [-1] * n
        child2 = [-1] * n
        
        # Copy segments
        child1[start:end+1] = p1_clean[start:end+1]
        child2[start:end+1] = p2_clean[start:end+1]
        
        # Fill remaining positions
        self._fill_crossover_child(child1, p2_clean, start, end)
        self._fill_crossover_child(child2, p1_clean, start, end)
        
        return child1, child2
    
    def _fill_crossover_child(self, child: List[int], parent: List[int], start: int, end: int):
        """Helper for crossover filling"""
        n = len(child)
        parent_idx = 0
        
        for i in range(n):
            if i < start or i > end:
                while parent[parent_idx] in child[start:end+1]:
                    parent_idx = (parent_idx + 1) % len(parent)
                child[i] = parent[parent_idx]
                parent_idx = (parent_idx + 1) % len(parent)
    
    def mutation_permutation(self, permutation: List[int], mutation_rate: float = 0.1) -> List[int]:
        """
        Mutation operators for permutation encoding
        """
        mutated = permutation.copy()
        
        if np.random.random() < mutation_rate:
            # Remove delimiters for mutation
            clean = [x for x in mutated if x != -1]
            
            # Apply mutation
            mutation_type = np.random.choice(['swap', 'insert', 'invert'])
            
            if mutation_type == 'swap' and len(clean) >= 2:
                i, j = np.random.choice(len(clean), 2, replace=False)
                clean[i], clean[j] = clean[j], clean[i]
            
            elif mutation_type == 'insert' and len(clean) >= 2:
                i = np.random.randint(len(clean))
                j = np.random.randint(len(clean))
                element = clean.pop(i)
                clean.insert(j, element)
            
            elif mutation_type == 'invert' and len(clean) >= 2:
                i, j = sorted(np.random.choice(len(clean), 2, replace=False))
                clean[i:j+1] = clean[i:j+1][::-1]
            
            # Reinsert delimiters randomly
            mutated = []
            for element in clean:
                mutated.append(element)
                if np.random.random() < 0.2:  # 20% chance of delimiter
                    mutated.append(-1)
        
        return mutated
    
    def local_search_binary(self, matrix: np.ndarray) -> np.ndarray:
        """
        Local search for binary matrix encoding
        """
        mutated = matrix.copy()
        
        # Random 2-opt move in binary representation
        n = len(mutated)
        
        # Find two edges to swap
        edges = []
        for i in range(n):
            for j in range(n):
                if mutated[i][j] == 1:
                    edges.append((i, j))
        
        if len(edges) >= 2:
            # Select two random edges
            edge1, edge2 = np.random.choice(len(edges), 2, replace=False)
            i1, j1 = edges[edge1]
            i2, j2 = edges[edge2]
            
            # Perform 2-opt swap
            mutated[i1][j1] = 0
            mutated[i2][j2] = 0
            mutated[i1][j2] = 1
            mutated[i2][j1] = 1
        
        return mutated

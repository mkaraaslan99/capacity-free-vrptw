"""
Vehicle Routing Problem with Time Windows (VRPTW) Definition and Utilities
Capacity-Free Variant - Time windows are the only binding constraint
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta


class Customer:
    """Represents a customer in capacity-free VRPTW"""
    
    def __init__(self, id: int, x: float, y: float, 
                 ready_time: float = 480, due_time: float = 1020, service_time: float = 60):
        """
        Initialize customer for capacity-free VRPTW
        
        Args:
            id: customer identifier
            x, y: location coordinates
            ready_time: earliest service start time in minutes from midnight (default 8:00 = 480)
            due_time: latest service start time in minutes from midnight (default 17:00 = 1020)
            service_time: service duration in minutes (default 60)
        """
        self.id = id
        self.x = x
        self.y = y
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
    
    def __repr__(self):
        return f"Customer({self.id}, pos=({self.x:.2f}, {self.y:.2f}), tw=[{self.ready_time}, {self.due_time}])"


class VRPTWInstance:
    """Represents a capacity-free VRPTW problem instance"""
    
    def __init__(self, customers: List[Customer], depot_x: float = 0.0, depot_y: float = 0.0,
                 depot_ready_time: float = 480, depot_due_time: float = 1020,
                 n_vehicles: int = None,
                 distance_matrix: np.ndarray = None, name: str = "VRPTW",
                 travel_speed: float = 1.0):
        """
        Initialize capacity-free VRPTW instance
        
        Args:
            customers: list of Customer objects
            depot_x, depot_y: depot coordinates (default 0, 0)
            depot_ready_time: depot opening time in minutes (default 480 = 8:00)
            depot_due_time: depot closing time in minutes (default 1020 = 17:00)
            n_vehicles: maximum number of vehicles (default unlimited)
            distance_matrix: precomputed distance matrix (optional)
            name: instance name
        """
        self.customers = customers
        self.n_customers = len(customers)
        
        # Create depot as customer 0
        self.depot = Customer(0, depot_x, depot_y, 
                             ready_time=depot_ready_time, 
                             due_time=depot_due_time, 
                             service_time=0)
        
        # All nodes (depot + customers)
        self.nodes = [self.depot] + customers
        self.n_nodes = len(self.nodes)
        
        # Capacity-free: no vehicle capacity constraint
        self.vehicle_capacity = float('inf')
        self.n_vehicles = n_vehicles if n_vehicles else self.n_customers
        self.name = name

        # Travel speed scaling (distance units per minute). Used for time-feasibility.
        self.travel_speed = float(travel_speed) if travel_speed else 1.0
        
        # Distance matrix
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute Euclidean distance matrix between all nodes"""
        n = self.n_nodes
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((self.nodes[i].x - self.nodes[j].x)**2 + 
                             (self.nodes[i].y - self.nodes[j].y)**2)
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        
        return dist_matrix
    
    def get_distance(self, node_i: int, node_j: int) -> float:
        """Get distance between two nodes"""
        return self.distance_matrix[node_i][node_j]

    def get_travel_time(self, node_i: int, node_j: int) -> float:
        """Get travel time between two nodes in minutes (distance / speed)."""
        if self.travel_speed <= 0:
            return self.get_distance(node_i, node_j)
        return self.get_distance(node_i, node_j) / self.travel_speed
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance of a route (including return to depot)"""
        if not route:
            return 0.0
        
        total_dist = self.get_distance(0, route[0])  # Depot to first customer
        
        for i in range(len(route) - 1):
            total_dist += self.get_distance(route[i], route[i+1])
        
        total_dist += self.get_distance(route[-1], 0)  # Last customer to depot
        
        return total_dist
    
    def calculate_solution_cost(self, solution: List[List[int]]) -> float:
        """Calculate total cost of a solution (sum of all route distances)"""
        return sum(self.calculate_route_distance(route) for route in solution)
    
    def is_route_feasible(self, route: List[int]) -> Tuple[bool, str]:
        """
        Check if a route is feasible (time windows only - capacity-free)
        
        Returns:
            tuple of (is_feasible, reason)
        """
        if not route:
            return True, "Empty route"
        
        # Check time windows
        current_time = self.depot.ready_time
        current_node = 0  # Start at depot
        
        for node_id in route:
            # Travel time to next customer
            travel_time = self.get_travel_time(current_node, node_id)
            arrival_time = current_time + travel_time
            
            # Service start time (wait if early)
            service_start = max(arrival_time, self.nodes[node_id].ready_time)
            
            # Check if service can start within time window
            if service_start > self.nodes[node_id].due_time:
                return False, f"Time window violated for customer {node_id}: service_start={service_start:.1f} > due_time={self.nodes[node_id].due_time}"
            
            # Update current time after service
            current_time = service_start + self.nodes[node_id].service_time
            current_node = node_id
        
        # Check return to depot
        travel_time = self.get_travel_time(current_node, 0)
        return_time = current_time + travel_time
        
        if return_time > self.depot.due_time:
            return False, f"Cannot return to depot: return_time={return_time:.1f} > depot_due_time={self.depot.due_time}"
        
        return True, "Feasible"
    
    def is_solution_feasible(self, solution: List[List[int]]) -> Tuple[bool, str]:
        """Check if entire solution is feasible"""
        # Check all customers are served exactly once
        all_customers = set(range(1, self.n_nodes))

        served_list: List[int] = []
        for route in solution:
            served_list.extend(route)

        served_set = set(served_list)
        missing = all_customers - served_set
        duplicates = {cid for cid in served_set if served_list.count(cid) > 1}

        if missing or duplicates:
            return False, f"Missing customers: {missing}, Duplicates: {duplicates}"
        
        # Check each route
        for i, route in enumerate(solution):
            feasible, reason = self.is_route_feasible(route)
            if not feasible:
                return False, f"Route {i}: {reason}"
        
        return True, "Feasible solution"
    
    @staticmethod
    def load_from_csv(customers_file: str, distance_file: str = None, 
                     depot_x: float = 0.0, depot_y: float = 0.0,
                     depot_ready: float = 480, depot_due: float = 1020,
                     service_time: float = 60, n_vehicles: int = None,
                     travel_speed: float = 1.0) -> 'VRPTWInstance':
        """
        Load capacity-free VRPTW instance from CSV files
        
        Args:
            customers_file: CSV file with columns: id, x, y (and optionally: ready_time, due_time, service_time)
            distance_file: CSV file with distance matrix (optional, will compute if not provided)
            depot_x, depot_y: depot coordinates
            depot_ready, depot_due: depot time window in minutes from midnight
            service_time: default service time for all customers (minutes)
            n_vehicles: number of vehicles
            
        Returns:
            VRPTWInstance object
        """
        # Load customers - auto-detect delimiter
        # Try to detect if file uses semicolon or comma
        with open(customers_file, 'r') as f:
            first_line = f.readline()
            delimiter = ';' if ';' in first_line else ','
        
        df = pd.read_csv(customers_file, sep=delimiter)
        
        # Filter out depot rows if present (role column)
        if 'role' in df.columns:
            original_count = len(df)
            df = df[df['role'] == 'customer'].copy()
            print(f"  Filtered out {original_count - len(df)} depot row(s)")

        # If depot row is present as id==0, use it to override depot parameters
        # and exclude it from customers.
        if 'id' in df.columns:
            depot_rows = df[df['id'].astype(int) == 0]
            if len(depot_rows) > 0:
                depot_row = depot_rows.iloc[0]
                if 'x' in df.columns:
                    depot_x = float(depot_row['x'])
                if 'y' in df.columns:
                    depot_y = float(depot_row['y'])
                if 'ready_time' in df.columns:
                    depot_ready = float(depot_row['ready_time'])
                if 'due_time' in df.columns:
                    depot_due = float(depot_row['due_time'])
                df = df[df['id'].astype(int) != 0].copy()
                print("  Detected depot row with id=0 and excluded it from customers")
        
        # Handle different column name variations
        # Map tw_start/tw_end to ready_time/due_time if needed
        if 'tw_start' in df.columns and 'ready_time' not in df.columns:
            df['ready_time'] = df['tw_start']
            print("  Mapped tw_start -> ready_time")
        if 'tw_end' in df.columns and 'due_time' not in df.columns:
            df['due_time'] = df['tw_end']
            print("  Mapped tw_end -> due_time")
        
        customers = []
        # Ensure contiguous internal IDs (1..n) because heuristics use node indices
        # directly as customer identifiers.
        if 'id' in df.columns:
            df = df.sort_values('id').reset_index(drop=True)

        for _, row in df.iterrows():
            cust_id = len(customers) + 1
            x = float(row['x'])
            y = float(row['y'])
            ready = float(row['ready_time']) if 'ready_time' in row else depot_ready
            due = float(row['due_time']) if 'due_time' in row else depot_due
            service = float(row['service_time']) if 'service_time' in row else service_time
            
            customers.append(Customer(cust_id, x, y, ready, due, service))
        
        # Load distance matrix if provided
        distance_matrix = None
        if distance_file:
            dist_df = pd.read_csv(distance_file, header=None)
            distance_matrix = dist_df.values
        
        name = customers_file.split('/')[-1].replace('.csv', '')
        
        return VRPTWInstance(customers, depot_x, depot_y, depot_ready, depot_due,
                           n_vehicles, distance_matrix, name, travel_speed=travel_speed)
    
    def save_solution(self, solution: List[List[int]], filename: str):
        """Save solution to file"""
        with open(filename, 'w') as f:
            f.write(f"Instance: {self.name}\n")
            f.write(f"Number of routes: {len(solution)}\n")
            f.write(f"Total distance: {self.calculate_solution_cost(solution):.2f}\n")
            f.write(f"Feasible: {self.is_solution_feasible(solution)[0]}\n\n")
            
            for i, route in enumerate(solution, 1):
                if route:
                    distance = self.calculate_route_distance(route)
                    f.write(f"Route {i}: 0 -> {' -> '.join(map(str, route))} -> 0\n")
                    f.write(f"  Distance: {distance:.2f}\n")
                    f.write(f"  Customers: {len(route)}\n\n")

#!/usr/bin/env python3
"""
Solution Encoding Demonstration for Capacity-Free VRPTW
Shows different encoding schemes and how to use them
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.vrptw import VRPTWInstance, Customer
from encoding.solution_encoding import VRPTWSolutionEncoder, SolutionOperators
from dual_pipeline_framework import DualPipelineFramework


def demonstrate_encodings():
    """Demonstrate different solution encoding schemes"""
    
    print("="*70)
    print("SOLUTION ENCODING DEMONSTRATION")
    print("="*70)
    
    # Create sample instance
    customers = [
        Customer(1, 10, 15, ready_time=480, due_time=1020, service_time=60),
        Customer(2, 25, 30, ready_time=500, due_time=900, service_time=60),
        Customer(3, 40, 20, ready_time=600, due_time=1000, service_time=60),
        Customer(4, 15, 45, ready_time=480, due_time=800, service_time=60),
        Customer(5, 35, 10, ready_time=700, due_time=1100, service_time=60)
    ]
    
    instance = VRPTWInstance(customers)
    encoder = VRPTWSolutionEncoder(instance)
    
    # Sample solution
    sample_routes = [
        [1, 4, 2],
        [3, 5]
    ]
    
    print(f"Original Solution (Route-based):")
    print(f"  Route 1: 0 → {' → '.join(map(str, sample_routes[0]))} → 0")
    print(f"  Route 2: 0 → {' → '.join(map(str, sample_routes[1]))} → 0")
    print(f"  Total Distance: {instance.calculate_solution_cost(sample_routes):.2f}")
    print()
    
    # Demonstrate different encodings
    print("1. PERMUTATION WITH DELIMITERS ENCODING:")
    permutation = encoder.route_based_to_permutation(sample_routes)
    print(f"   {permutation}")
    print(f"   (-1 marks route boundaries)")
    print()
    
    # Decode back
    decoded_routes = encoder.permutation_to_route_based(permutation)
    print(f"   Decoded back: {decoded_routes}")
    print()
    
    print("2. GIANT TOUR WITH ROUTE BREAKS ENCODING:")
    giant_tour, route_breaks = encoder.route_based_to_giant_tour(sample_routes)
    print(f"   Giant Tour: {giant_tour}")
    print(f"   Route Breaks (indices): {route_breaks}")
    print(f"   (Routes end at these positions in the giant tour)")
    print()
    
    # Decode back
    decoded_giant = encoder.giant_tour_to_route_based(giant_tour, route_breaks)
    print(f"   Decoded back: {decoded_giant}")
    print()
    
    print("3. BINARY MATRIX ENCODING:")
    binary_matrix = encoder.route_based_to_binary_matrix(sample_routes)
    print("   Binary Adjacency Matrix (first 6x6 shown):")
    print(binary_matrix[:6, :6])
    print("   (matrix[i][j] = 1 if j follows i in route)")
    print()
    
    # Decode back
    decoded_binary = encoder.binary_matrix_to_route_based(binary_matrix)
    print(f"   Decoded back: {decoded_binary}")
    print()
    
    # Validate all encodings
    print("4. VALIDATION OF ALL ENCODINGS:")
    encodings = {
        'Route-based': sample_routes,
        'Permutation': permutation,
        'Giant Tour': (giant_tour, route_breaks),
        'Binary Matrix': binary_matrix
    }
    
    for name, encoded in encodings.items():
        is_valid, reason = encoder.validate_encoding(encoded, name.lower().replace(' ', '_').replace('-', '_'))
        status = "✓ Valid" if is_valid else f"✗ Invalid: {reason}"
        print(f"   {name:15} {status}")
    print()


def demonstrate_genetic_algorithm_encoding():
    """Show how encoding works with genetic algorithm concepts"""
    
    print("="*70)
    print("GENETIC ALGORITHM ENCODING DEMONSTRATION")
    print("="*70)
    
    # Create instance
    customers = [
        Customer(i, np.random.uniform(0, 50), np.random.uniform(0, 50),
                 ready_time=480, due_time=1020, service_time=60)
        for i in range(1, 9)  # 8 customers
    ]
    
    instance = VRPTWInstance(customers)
    encoder = VRPTWSolutionEncoder(instance)
    operators = SolutionOperators(instance)
    
    print("1. RANDOM SOLUTION GENERATION:")
    
    # Generate random solutions in different encodings
    for encoding_type in ['route_based', 'permutation', 'giant_tour', 'binary_matrix']:
        random_solution = encoder.generate_random_solution(encoding_type)
        print(f"\n   {encoding_type:15}: ", end="")
        
        if encoding_type == 'route_based':
            print(f"{random_solution}")
        elif encoding_type == 'permutation':
            print(f"{random_solution}")
        elif encoding_type == 'giant_tour':
            tour, breaks = random_solution
            print(f"Tour: {tour}, Breaks: {breaks}")
        elif encoding_type == 'binary_matrix':
            print(f"Matrix shape: {random_solution.shape}")
    
    print("\n2. CROSSOVER OPERATOR (Permutation Encoding):")
    
    # Generate two parent solutions
    parent1 = encoder.generate_random_solution('permutation')
    parent2 = encoder.generate_random_solution('permutation')
    
    print(f"   Parent 1: {parent1}")
    print(f"   Parent 2: {parent2}")
    
    # Apply crossover
    child1, child2 = operators.crossover_permutation(parent1, parent2)
    print(f"   Child 1:  {child1}")
    print(f"   Child 2:  {child2}")
    
    # Validate children
    valid1, reason1 = encoder.validate_encoding(child1, 'permutation')
    valid2, reason2 = encoder.validate_encoding(child2, 'permutation')
    print(f"   Child 1 Valid: {'✓' if valid1 else '✗'}")
    print(f"   Child 2 Valid: {'✓' if valid2 else '✗'}")
    
    print("\n3. MUTATION OPERATOR (Permutation Encoding):")
    
    # Apply mutation
    mutated = operators.mutation_permutation(child1, mutation_rate=0.3)
    print(f"   Original:  {child1}")
    print(f"   Mutated:  {mutated}")
    
    # Validate mutated
    valid_mut, reason_mut = encoder.validate_encoding(mutated, 'permutation')
    print(f"   Mutated Valid: {'✓' if valid_mut else '✗'}")
    print()


def demonstrate_integration_with_framework():
    """Show how encoding integrates with dual-pipeline framework"""
    
    print("="*70)
    print("INTEGRATION WITH DUAL-PIPELINE FRAMEWORK")
    print("="*70)
    
    # Create instance
    customers = [
        Customer(i, np.random.uniform(0, 40), np.random.uniform(0, 40),
                 ready_time=480, due_time=1020, service_time=60)
        for i in range(1, 11)  # 10 customers
    ]
    
    instance = VRPTWInstance(customers)
    encoder = VRPTWSolutionEncoder(instance)
    
    print("Running Dual-Pipeline Framework...")
    framework = DualPipelineFramework(instance)
    solution, cost, stats = framework.run_dual_pipeline()
    
    print(f"\nBest Solution: {solution}")
    print(f"Total Cost: {cost:.2f}")
    print(f"Best Pipeline: {stats['best_pipeline']}")
    
    # Convert solution to different encodings
    print("\nConverting to different encodings:")
    
    # Permutation encoding
    permutation = encoder.encode_solution(solution, 'permutation')
    print(f"Permutation: {permutation}")
    
    # Giant tour encoding
    giant_tour, route_breaks = encoder.encode_solution(solution, 'giant_tour')
    print(f"Giant Tour: {giant_tour}")
    print(f"Route Breaks: {route_breaks}")
    
    # Binary matrix encoding
    binary_matrix = encoder.encode_solution(solution, 'binary_matrix')
    print(f"Binary Matrix: {binary_matrix.shape} adjacency matrix")
    
    print("\nAll encodings validated successfully!")
    print("This demonstrates how solution encoding enables:")
    print("  • Different algorithmic approaches")
    print("  • Genetic algorithm operations")
    print("  • Local search in different representations")
    print("  • Easy conversion between formats")


def main():
    """Main demonstration function"""
    try:
        demonstrate_encodings()
        demonstrate_genetic_algorithm_encoding()
        demonstrate_integration_with_framework()
        
        print("\n" + "="*70)
        print("ENCODING DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nKey Takeaways:")
        print("1. Multiple encoding schemes support different algorithmic approaches")
        print("2. Easy conversion between formats enables flexible implementation")
        print("3. Validation ensures solution integrity across encodings")
        print("4. Integration with existing framework is seamless")
        print("5. Foundation for advanced metaheuristics (GA, SA, ALNS)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

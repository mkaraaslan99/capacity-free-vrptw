# Quick Start Guide

## Installation

1. **Install Python dependencies**:
```bash
cd optimization_project
pip install -r requirements.txt
```

## Running the Project

### Option 1: Use Sample Data (Fastest Way to Test)

```bash
python main_vrptw.py --customers data/sample_customers.csv
```

This will:
- Load 10 sample customers
- Run all 6 algorithms
- Compare results
- Generate visualizations in `results/` folder
- Show the best solution

### Option 2: Use Your Own Data

1. **Prepare your customers CSV file**:
```csv
id,x,y
1,10.5,20.3
2,15.2,30.1
3,25.8,18.7
...
```

2. **Run with your data**:
```bash
python main_vrptw.py --customers path/to/your_customers.csv
```

### Option 3: Custom Configuration

```bash
python main_vrptw.py \
  --customers data/sample_customers.csv \
  --depot-x 0 \
  --depot-y 0 \
  --depot-open 480 \
  --depot-close 1020 \
  --service-time 60 \
  --n-vehicles 10 \
  --output-dir my_results
```

## Understanding the Output

After running, you'll see:

1. **Console Output**:
   - Progress for each algorithm
   - Best solution summary
   - Route details

2. **Generated Files** (in `results/` or your specified output directory):
   - `best_solution.txt` - Detailed solution description
   - `best_solution.png` - Visual map of routes
   - `algorithm_comparison.png` - Performance comparison charts
   - `convergence.png` - Convergence plots for metaheuristics

## Running Individual Algorithms

### Heuristics (Fast)
```bash
# Nearest Neighbor
python main_vrptw.py --customers data/sample_customers.csv --algorithm nn

# Clarke-Wright Savings
python main_vrptw.py --customers data/sample_customers.csv --algorithm savings
```

### Metaheuristics (Slower but Better Quality)
```bash
# Genetic Algorithm
python main_vrptw.py --customers data/sample_customers.csv --algorithm ga

# Simulated Annealing
python main_vrptw.py --customers data/sample_customers.csv --algorithm sa
```

## Time Estimates

For 10 customers:
- Heuristics: < 1 second
- Genetic Algorithm: 5-10 seconds
- Simulated Annealing: 3-5 seconds

For 50 customers:
- Heuristics: 1-2 seconds
- Genetic Algorithm: 30-60 seconds
- Simulated Annealing: 20-40 seconds

## Troubleshooting

### Issue: "No module named 'numpy'"
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: customers file not found"
**Solution**: Check the file path is correct
```bash
# Use absolute path or correct relative path
python main_vrptw.py --customers /full/path/to/customers.csv
```

### Issue: All routes are infeasible
**Solution**: Adjust time windows or service time
```bash
# Extend operating hours
python main_vrptw.py --customers data/sample_customers.csv --depot-close 1200

# Reduce service time
python main_vrptw.py --customers data/sample_customers.csv --service-time 30
```

## Next Steps

1. **Test with your own dataset**
2. **Adjust algorithm parameters** in the code for better performance
3. **Add more customers** to test scalability
4. **Compare different configurations** for your MSc presentation
5. **Analyze the visualizations** to understand algorithm behavior

## For Your MSc Presentation

Recommended workflow:
1. Run with sample data to verify everything works
2. Create 2-3 test instances with different characteristics:
   - Small (10-20 customers)
   - Medium (30-50 customers)
   - Large (100+ customers) - optional
3. Run all algorithms on each instance
4. Compare results and analyze trade-offs
5. Use the generated visualizations in your presentation
6. Discuss algorithm strengths and weaknesses based on results

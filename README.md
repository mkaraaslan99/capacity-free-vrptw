# Capacity-Free VRPTW: Dual-Pipeline Heuristic Framework

A comprehensive implementation of heuristic algorithms for the Vehicle Routing Problem with Time Windows (VRPTW), focusing on capacity-free scenarios where temporal constraints are the primary concern.

## 📋 Project Overview

This project implements a dual-pipeline heuristic framework that combines classical construction heuristics (Nearest Neighbor and Clarke-Wright Savings) with local search operators (2-opt and relocation) to solve capacity-free VRPTW instances. The framework is designed for academic research and provides detailed stage-by-stage analysis of algorithm performance.

## 🎯 Key Features

- **Dual-Pipeline Architecture**: Runs NN and CW pipelines in parallel, selecting the best solution
- **Time-Window Focused**: Removes capacity constraints to isolate temporal optimization effects
- **Controlled Randomness**: RCL-based construction with seeded random number generation
- **Urgency-Aware Variant**: Weighted-NN incorporating time-window slack normalization
- **Comprehensive Reporting**: Stage-by-stage performance metrics with statistical analysis
- **Academic LaTeX Report**: Complete research paper with detailed methodology and results

## 📁 Project Structure

```
optimization_project/
├── heuristics/              # Core heuristic implementations
│   ├── vrptw_nearest_neighbor.py
│   ├── vrptw_savings_new.py
│   └── ...
├── local_search/            # Local search operators
│   ├── two_opt.py
│   └── relocation.py
├── metaheuristics/          # Advanced optimization algorithms
│   ├── genetic_algorithm.py
│   └── simulated_annealing.py
├── problems/                # Problem instance definitions
├── utils/                   # Utility functions
├── data/                    # Sample datasets (Solomon C1)
├── report_outputs/          # Generated tables and figures
├── main_capacity_free_vrptw.py    # Main execution script
├── run_all_reports.py       # Generate all experimental results
├── vrptw_complete_report.tex      # Full LaTeX research paper
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages: numpy, matplotlib, pandas

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/capacity-free-vrptw.git
cd capacity-free-vrptw

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run capacity-free VRPTW with default settings
python main_capacity_free_vrptw.py --customers data/solomon_c101_10.csv

# Run with Weighted-NN variant
python main_capacity_free_vrptw.py --customers data/solomon_c101_10.csv \
    --urgency-weight 0.5

# Generate all experimental results
python run_all_reports.py
```

## 📊 Experimental Results

The framework has been tested on Solomon C1 benchmark instances with:
- **Instance sizes**: 10 and 20 customers
- **Time-window configurations**: Homogeneous and heterogeneous
- **Repeated runs**: 10 independent runs per configuration
- **RCL size**: k=3 for controlled randomness

### Key Findings

- **CW outperforms NN**: 10-20% lower initial costs
- **Local search is essential**: 5-12% cost reduction
- **High consistency**: CV values below 4% across all algorithms
- **Dual-pipeline robustness**: Best stability with 2.32-3.02% CV

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Detailed usage guide with examples
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Technical implementation details
- **[vrptw_complete_report.tex](vrptw_complete_report.tex)**: Full academic paper (compile with pdflatex)

## 🔬 Research Paper

The complete research methodology, experimental setup, and results are documented in the LaTeX report:

```bash
# Compile the LaTeX report
cd optimization_project
pdflatex vrptw_complete_report.tex
pdflatex vrptw_complete_report.tex  # Run twice for references
```

## 🛠️ Algorithm Components

### Construction Heuristics
- **Nearest Neighbor (NN)**: Sequential insertion with optional urgency weighting
- **Clarke-Wright Savings (CW)**: Route merging with savings-based selection
- **RCL Mechanism**: Top-k candidate selection for controlled randomness

### Local Search Operators
- **2-opt**: Intra-route segment reversal
- **Relocation**: Intra- and inter-route customer moves
- **Feasibility Preservation**: All moves maintain time-window constraints

### Dual-Pipeline Framework
1. NN pipeline: NN_initial → 2-opt → Relocation → NN_improved
2. CW pipeline: CW_initial → 2-opt → Relocation → CW_improved
3. Selection: Choose best improved solution

## 📈 Performance Metrics

The framework reports the following metrics at each stage:
- Total travel cost (mean ± std)
- Number of routes/vehicles
- Feasibility status (100% maintained)
- Runtime (seconds)
- Coefficient of variation (consistency measure)

## 🎓 Academic Context

This project was developed for MSc coursework in Industrial Engineering at Hacettepe University. The methodology is based on classical VRPTW literature:

- Solomon (1987): Benchmark instances and baseline heuristics
- Bräysy & Gendreau (2005): VRPTW algorithm survey
- Feo & Resende (1995): GRASP framework with RCL
- Gomes & Selman (2001): Algorithm portfolios

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{karaaslan2025vrptw,
  title={Time-Windowed Vehicle Routing Without Capacity Constraints: A Dual-Pipeline Heuristic Framework},
  author={Karaaslan, Mert},
  year={2025},
  school={Hacettepe University},
  type={MSc Project Report}
}
```

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

**Mert Karaaslan**  
Department of Industrial Engineering  
Hacettepe University

## 📄 License

This project is available for academic and educational purposes. Please contact the author for commercial use.

## 🙏 Acknowledgments

- Solomon benchmark instances for VRPTW research
- Hacettepe University Industrial Engineering Department
- Classical VRPTW literature and methodology

---

**Note**: This framework is designed for research and educational purposes. For production use, consider additional optimizations and robustness enhancements.

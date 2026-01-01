#!/usr/bin/env python3
"""One-shot orchestrator to generate all report artifacts.

Runs (in order):
1) Homogeneous instances: detailed runs + tables + figures
2) Generate heterogeneous time-window datasets
3) Heterogeneous instances baseline (NN urgency disabled): detailed runs + tables + figures
4) Heterogeneous instances weighted NN (urgency enabled): detailed runs + tables + figures

All outputs are written under optimization_project/report_outputs/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(args: list[str]) -> None:
    print("\n$ " + " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    # 1) Homogeneous
    run([
        PY,
        "detailed_runs_report.py",
        "--rcl-size",
        "3",
        "--base-seed",
        "42",
        "--travel-speed",
        "5.0",
        "--max-route-duration",
        "480",
        "--nn-w-distance",
        "1.0",
        "--nn-w-urgency",
        "0.0",
        "--out",
        "report_outputs/vrptw_detailed_runs.csv",
    ])

    run([
        PY,
        "export_report_tables.py",
        "--in",
        "report_outputs/vrptw_detailed_runs.csv",
        "--out-dir",
        "report_outputs/tables",
    ])

    run([
        PY,
        "plot_detailed_runs_figures.py",
        "--csv",
        "report_outputs/vrptw_detailed_runs.csv",
        "--out-dir",
        "report_outputs/figures",
    ])

    # 2) Generate hetero datasets
    run([
        PY,
        "generate_heterogeneous_tw.py",
        "--in-files",
        "data/customers_10_vrptw.csv",
        "data/customers_20_vrptw.csv",
        "--out-dir",
        "data",
        "--suffix",
        "_hetero_tw",
    ])

    hetero_10 = "data/customers_10_vrptw_hetero_tw.csv"
    hetero_20 = "data/customers_20_vrptw_hetero_tw.csv"

    # 3) Hetero baseline
    run([
        PY,
        "detailed_runs_report.py",
        "--case",
        f"{hetero_10}:5",
        "--case",
        f"{hetero_20}:10",
        "--rcl-size",
        "3",
        "--base-seed",
        "42",
        "--travel-speed",
        "5.0",
        "--max-route-duration",
        "480",
        "--nn-w-distance",
        "1.0",
        "--nn-w-urgency",
        "0.0",
        "--out",
        "report_outputs/vrptw_detailed_runs_hetero_baseline.csv",
    ])

    run([
        PY,
        "export_report_tables.py",
        "--in",
        "report_outputs/vrptw_detailed_runs_hetero_baseline.csv",
        "--out-dir",
        "report_outputs/tables_hetero_baseline",
    ])

    run([
        PY,
        "plot_detailed_runs_figures.py",
        "--csv",
        "report_outputs/vrptw_detailed_runs_hetero_baseline.csv",
        "--out-dir",
        "report_outputs/figures_hetero_baseline",
    ])

    # 4) Hetero weighted NN
    run([
        PY,
        "detailed_runs_report.py",
        "--case",
        f"{hetero_10}:5",
        "--case",
        f"{hetero_20}:10",
        "--rcl-size",
        "3",
        "--base-seed",
        "42",
        "--travel-speed",
        "5.0",
        "--max-route-duration",
        "480",
        "--nn-w-distance",
        "1.0",
        "--nn-w-urgency",
        "2.0",
        "--out",
        "report_outputs/vrptw_detailed_runs_hetero_weightedNN_u2.csv",
    ])

    run([
        PY,
        "export_report_tables.py",
        "--in",
        "report_outputs/vrptw_detailed_runs_hetero_weightedNN_u2.csv",
        "--out-dir",
        "report_outputs/tables_hetero_weightedNN_u2",
    ])

    run([
        PY,
        "plot_detailed_runs_figures.py",
        "--csv",
        "report_outputs/vrptw_detailed_runs_hetero_weightedNN_u2.csv",
        "--out-dir",
        "report_outputs/figures_hetero_weightedNN_u2",
    ])

    print("\nAll artifacts generated under report_outputs/.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export clean report-ready tables (wide per-run + summary) from vrptw_detailed_runs.csv to both CSV and Excel formats."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export pivoted report tables from VRPTW detailed runs CSV")
    ap.add_argument("--in", dest="inp", default="report_outputs/vrptw_detailed_runs.csv")
    ap.add_argument("--out-dir", default="report_outputs/tables")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # Create method key
    df["method"] = df["variant"].astype(str) + "_" + df["stage"].astype(str)

    # Per-run wide tables for COST
    wide_cost = (
        df.pivot_table(
            index=["instance", "run_idx", "seed"],
            columns="method",
            values="cost",
            aggfunc="first",
        )
        .reset_index()
    )

    # Per-run wide tables for ROUTES
    wide_routes = (
        df.pivot_table(
            index=["instance", "run_idx", "seed"],
            columns="method",
            values="routes",
            aggfunc="first",
        )
        .reset_index()
    )

    # Summary stats per method per instance
    summary = (
        df.groupby(["instance", "method"], as_index=False)
        .agg(
            n_runs=("run_idx", "count"),
            feasible_rate=("feasible", "mean"),
            cost_mean=("cost", "mean"),
            cost_std=("cost", "std"),
            cost_min=("cost", "min"),
            cost_max=("cost", "max"),
            routes_mean=("routes", "mean"),
            time_mean_s=("time_s", "mean"),
            time_std_s=("time_s", "std"),
        )
    )
    summary["feasible_rate"] = 100.0 * summary["feasible_rate"]

    # CSV paths
    wide_cost_csv = out_dir / "vrptw_per_run_cost_wide.csv"
    wide_routes_csv = out_dir / "vrptw_per_run_routes_wide.csv"
    summary_csv = out_dir / "vrptw_summary_stats.csv"
    
    # Excel paths
    wide_cost_xlsx = out_dir / "vrptw_per_run_cost_wide.xlsx"
    wide_routes_xlsx = out_dir / "vrptw_per_run_routes_wide.xlsx"
    summary_xlsx = out_dir / "vrptw_summary_stats.xlsx"
    excel_combined = out_dir / "vrptw_all_results.xlsx"

    # Save CSV files
    wide_cost.to_csv(wide_cost_csv, index=False)
    wide_routes.to_csv(wide_routes_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    
    # Save individual Excel files
    wide_cost.to_excel(wide_cost_xlsx, index=False, sheet_name="Cost_Per_Run")
    wide_routes.to_excel(wide_routes_xlsx, index=False, sheet_name="Routes_Per_Run")
    summary.to_excel(summary_xlsx, index=False, sheet_name="Summary_Stats")
    
    # Save combined Excel file with multiple sheets
    with pd.ExcelWriter(excel_combined, engine='openpyxl') as writer:
        wide_cost.to_excel(writer, sheet_name='Cost_Per_Run', index=False)
        wide_routes.to_excel(writer, sheet_name='Routes_Per_Run', index=False)
        summary.to_excel(writer, sheet_name='Summary_Stats', index=False)
        df.to_excel(writer, sheet_name='Raw_Data', index=False)

    print("Wrote report tables (CSV):")
    print(f"- {wide_cost_csv}")
    print(f"- {wide_routes_csv}")
    print(f"- {summary_csv}")
    print("\nWrote report tables (Excel):")
    print(f"- {wide_cost_xlsx}")
    print(f"- {wide_routes_xlsx}")
    print(f"- {summary_xlsx}")
    print(f"- {excel_combined} (combined with all sheets)")


if __name__ == "__main__":
    main()

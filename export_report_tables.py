#!/usr/bin/env python3
"""Export clean report-ready tables (wide per-run + summary) from vrptw_detailed_runs.csv."""

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

    wide_cost_path = out_dir / "vrptw_per_run_cost_wide.csv"
    wide_routes_path = out_dir / "vrptw_per_run_routes_wide.csv"
    summary_path = out_dir / "vrptw_summary_stats.csv"

    wide_cost.to_csv(wide_cost_path, index=False)
    wide_routes.to_csv(wide_routes_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Wrote report tables:")
    print(f"- {wide_cost_path}")
    print(f"- {wide_routes_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()

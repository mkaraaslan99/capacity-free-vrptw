#!/usr/bin/env python3
"""Create publication-ready figures from report_outputs/vrptw_detailed_runs.csv.

Generates per-instance plots for cost / routes / runtime across variants:
- NN initial, NN improved
- CW initial, CW improved
- Dual final

Output: optimization_project/report_outputs/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ORDER = [
    ("NN", "initial", "NN initial"),
    ("NN", "improved", "NN improved"),
    ("CW", "initial", "CW initial"),
    ("CW", "improved", "CW improved"),
    ("Dual", "final", "Final (Dual)"),
]


def _add_method_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {(v, s): label for v, s, label in ORDER}
    df["method"] = [mapping.get((v, s), f"{v}_{s}") for v, s in zip(df["variant"], df["stage"]) ]
    df["method"] = pd.Categorical(df["method"], categories=[x[2] for x in ORDER], ordered=True)
    return df


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_instance(df_inst: pd.DataFrame, out_dir: Path) -> list[Path]:
    df_inst = _add_method_label(df_inst)
    instance_name = str(df_inst["instance"].iloc[0])

    paths: list[Path] = []

    # Figure 1: Distribution (box + points) for cost
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    sns.boxplot(data=df_inst, x="method", y="cost", ax=ax, showfliers=False)
    sns.stripplot(data=df_inst, x="method", y="cost", ax=ax, color="black", size=4, jitter=0.15, alpha=0.6)
    ax.set_title(f"{instance_name} | Cost distribution across runs")
    ax.set_xlabel("")
    ax.set_ylabel("Total distance (cost)")
    ax.tick_params(axis="x", rotation=20)
    p = out_dir / f"{instance_name}_cost_distribution.png"
    _save(fig, p)
    paths.append(p)

    # Figure 2: Mean ± std for cost (bar + errorbar)
    agg = (
        df_inst.groupby("method", observed=True)
        .agg(cost_mean=("cost", "mean"), cost_std=("cost", "std"), time_mean=("time_s", "mean"), routes_mean=("routes", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.bar(agg["method"].astype(str), agg["cost_mean"], yerr=agg["cost_std"].fillna(0.0), capsize=6)
    ax.set_title(f"{instance_name} | Mean cost ± std")
    ax.set_xlabel("")
    ax.set_ylabel("Total distance (cost)")
    ax.tick_params(axis="x", rotation=20)
    p = out_dir / f"{instance_name}_cost_mean_std.png"
    _save(fig, p)
    paths.append(p)

    # Figure 3: Routes distribution
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    sns.boxplot(data=df_inst, x="method", y="routes", ax=ax, showfliers=False)
    sns.stripplot(data=df_inst, x="method", y="routes", ax=ax, color="black", size=4, jitter=0.15, alpha=0.6)
    ax.set_title(f"{instance_name} | #Routes distribution across runs")
    ax.set_xlabel("")
    ax.set_ylabel("Number of routes")
    ax.tick_params(axis="x", rotation=20)
    p = out_dir / f"{instance_name}_routes_distribution.png"
    _save(fig, p)
    paths.append(p)

    # Figure 4: Runtime mean ± std
    agg_t = (
        df_inst.groupby("method", observed=True)
        .agg(time_mean=("time_s", "mean"), time_std=("time_s", "std"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.bar(agg_t["method"].astype(str), agg_t["time_mean"], yerr=agg_t["time_std"].fillna(0.0), capsize=6)
    ax.set_title(f"{instance_name} | Mean runtime ± std")
    ax.set_xlabel("")
    ax.set_ylabel("Runtime (s)")
    ax.tick_params(axis="x", rotation=20)
    p = out_dir / f"{instance_name}_runtime_mean_std.png"
    _save(fig, p)
    paths.append(p)

    return paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot VRPTW detailed run figures")
    ap.add_argument(
        "--csv",
        default="report_outputs/vrptw_detailed_runs.csv",
        help="Input CSV from detailed_runs_report.py",
    )
    ap.add_argument(
        "--out-dir",
        default="report_outputs/figures",
        help="Output directory for figures",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(csv_path)

    # Basic sanity: keep only the methods we care about
    keep = {(v, s) for v, s, _ in ORDER}
    df = df[df.apply(lambda r: (r["variant"], r["stage"]) in keep, axis=1)].copy()

    # Seaborn style
    sns.set_theme(style="whitegrid")

    all_paths: list[Path] = []
    for instance_name, df_inst in df.groupby("instance", sort=True):
        all_paths.extend(plot_instance(df_inst, out_dir))

    print("Generated figures:")
    for p in all_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()

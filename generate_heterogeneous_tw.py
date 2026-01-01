#!/usr/bin/env python3
"""Generate heterogeneous time-window versions of existing VRPTW CSV datasets.

Purpose: create Solomon-like variability so urgency-based Weighted-NN has an effect.

Strategy (per customer id > 0):
- due_time decreases with id modulo a cycle
- window_width alternates between tight and wide windows
- ready_time = max(global_min_ready, due_time - window_width)

The depot row (id=0) is preserved as-is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_heterogeneous(df: pd.DataFrame, *, min_ready: float = 540.0, depot_due: float = 1020.0) -> pd.DataFrame:
    df2 = df.copy()

    # Only modify customers (exclude depot row if present)
    mask = df2["id"].astype(int) != 0

    # Build due times with moderate spread; keep within [min_ready + service, depot_due]
    ids = df2.loc[mask, "id"].astype(int)

    # due_time pattern: 1020, 990, 960, 930, 900, then repeats
    # (keeps windows heterogeneous but avoids making many customers infeasible)
    due = depot_due - (ids % 5) * 30.0

    # window width alternates (tight vs wide)
    width = ids.apply(lambda x: 240.0 if (x % 2 == 0) else 360.0).astype(float)

    ready = (due - width).clip(lower=min_ready)

    # Ensure due >= ready + 60 (service) for all
    service = df2.loc[mask, "service_time"].astype(float)
    min_due = ready + service
    due = due.where(due >= min_due, min_due)

    df2.loc[mask, "ready_time"] = ready
    df2.loc[mask, "due_time"] = due

    return df2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate heterogeneous time-window VRPTW datasets")
    ap.add_argument("--in-files", nargs="+", required=True, help="Input VRPTW CSV files")
    ap.add_argument("--out-dir", default="data", help="Output directory (default: data)")
    ap.add_argument("--suffix", default="_hetero_tw", help="Suffix added to output filename stem")
    ap.add_argument("--min-ready", type=float, default=540.0)
    ap.add_argument("--depot-due", type=float, default=1020.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path_str in args.in_files:
        in_path = Path(in_path_str)
        df = pd.read_csv(in_path)
        df2 = make_heterogeneous(df, min_ready=args.min_ready, depot_due=args.depot_due)

        out_path = out_dir / f"{in_path.stem}{args.suffix}{in_path.suffix}"
        df2.to_csv(out_path, index=False)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

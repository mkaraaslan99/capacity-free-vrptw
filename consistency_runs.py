#!/usr/bin/env python3
"""Run each VRPTW instance multiple times and report variance/consistency stats.

This script is intended for reporting algorithm robustness. It runs:
- NN (initial)
- NN + Local Search (2-opt + relocation) + duration splitting (improved)
- CW (initial)
- CW + Local Search (2-opt + relocation) + duration splitting (improved)
- Feasible Dual-Pipeline (final)

Outputs mean/std/var/min/max and feasible rate per metric.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from problems.vrptw import VRPTWInstance
from heuristics.vrptw_nearest_neighbor import nearest_neighbor_vrptw
from heuristics.vrptw_savings_new import clarke_wright_savings_vrptw
from final_feasible_pipeline import final_feasible_dual_pipeline
from local_search.local_search_operators import combined_local_search

# Reuse the shared route-duration post-processing from the CLI runner to keep comparisons consistent.
from main_capacity_free_vrptw import _apply_max_route_duration


@dataclass(frozen=True)
class RunRecord:
    name: str
    cost: float
    routes: int
    feasible: bool
    wall_time_s: float


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "var": float("nan"), "min": float("nan"), "max": float("nan")}

    mean = statistics.fmean(values)
    var = statistics.pvariance(values) if len(values) > 1 else 0.0
    std = var**0.5
    return {
        "mean": mean,
        "std": std,
        "var": var,
        "min": min(values),
        "max": max(values),
    }


def _run_initial_and_improved(
    vrptw_instance: VRPTWInstance,
    algo: str,
    max_vehicles: Optional[int],
    max_route_duration: Optional[float],
    rcl_size: int,
    seed: Optional[int],
    nn_w_distance: float,
    nn_w_urgency: float,
) -> Tuple[RunRecord, RunRecord]:
    """Return (initial, improved) for NN or CW."""

    t0 = time.time()
    if algo == "nn":
        sol, cost, _ = nearest_neighbor_vrptw(
            vrptw_instance,
            max_vehicles,
            rcl_size=rcl_size,
            seed=seed,
            rng=random.Random(seed) if seed is not None else None,
            w_distance=nn_w_distance,
            w_urgency=nn_w_urgency,
        )
        name = "NN"
    elif algo == "cw":
        sol, cost, _ = clarke_wright_savings_vrptw(
            vrptw_instance,
            max_vehicles,
            rcl_size=rcl_size,
            seed=seed,
            rng=random.Random(seed) if seed is not None else None,
        )
        name = "CW"
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    sol = _apply_max_route_duration(vrptw_instance, sol, max_route_duration)
    cost = vrptw_instance.calculate_solution_cost(sol)
    feasible = vrptw_instance.is_solution_feasible(sol)[0]
    initial = RunRecord(name=f"{name}_initial", cost=cost, routes=len(sol), feasible=feasible, wall_time_s=time.time() - t0)

    t1 = time.time()
    sol2, _, _ = combined_local_search(vrptw_instance, [r.copy() for r in sol])
    sol2 = _apply_max_route_duration(vrptw_instance, sol2, max_route_duration)
    cost2 = vrptw_instance.calculate_solution_cost(sol2)
    feasible2 = vrptw_instance.is_solution_feasible(sol2)[0]
    improved = RunRecord(name=f"{name}_improved", cost=cost2, routes=len(sol2), feasible=feasible2, wall_time_s=time.time() - t1)

    return initial, improved


def _run_feasible_dual(
    vrptw_instance: VRPTWInstance,
    max_vehicles: Optional[int],
    max_route_duration: Optional[float],
    rcl_size: int,
    seed: Optional[int],
    nn_w_distance: float,
    nn_w_urgency: float,
) -> RunRecord:
    t0 = time.time()

    # Avoid double-splitting inside the pipeline; apply shared splitting afterwards.
    internal_max_route_duration = 10**9
    with contextlib.redirect_stdout(io.StringIO()):
        sol, _, _ = final_feasible_dual_pipeline(
            vrptw_instance,
            max_vehicles=max_vehicles,
            nn_variant="sequential",
            cw_variant="standard",
            max_route_duration=internal_max_route_duration,
            nn_rcl_size=rcl_size,
            cw_rcl_size=rcl_size,
            seed=seed,
            nn_w_distance=nn_w_distance,
            nn_w_urgency=nn_w_urgency,
        )

    sol = _apply_max_route_duration(vrptw_instance, sol, max_route_duration)
    cost = vrptw_instance.calculate_solution_cost(sol)
    feasible = vrptw_instance.is_solution_feasible(sol)[0]

    return RunRecord(
        name="FeasibleDual_final",
        cost=cost,
        routes=len(sol),
        feasible=feasible,
        wall_time_s=time.time() - t0,
    )


def _print_summary_table(instance_name: str, records: List[RunRecord]) -> None:
    by_name: Dict[str, List[RunRecord]] = {}
    for r in records:
        by_name.setdefault(r.name, []).append(r)

    print("\n" + "=" * 90)
    print(f"CONSISTENCY SUMMARY | {instance_name}")
    print("=" * 90)
    header = (
        f"{'Variant':24s} | {'feas%':>6s} | {'cost_mean':>10s} | {'cost_std':>9s} | {'cost_min':>10s} | {'cost_max':>10s} | {'time_mean(s)':>11s}"
    )
    print(header)
    print("-" * len(header))

    for name in sorted(by_name.keys()):
        rs = by_name[name]
        costs = [x.cost for x in rs]
        times = [x.wall_time_s for x in rs]
        cost_stats = _summarize(costs)
        time_stats = _summarize(times)
        feas_rate = 100.0 * (sum(1 for x in rs if x.feasible) / len(rs))

        print(
            f"{name:24s} | {feas_rate:6.1f} | {cost_stats['mean']:10.2f} | {cost_stats['std']:9.2f} | {cost_stats['min']:10.2f} | {cost_stats['max']:10.2f} | {time_stats['mean']:11.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VRPTW instances multiple times and compute variance stats")

    parser.add_argument(
        "--customers",
        nargs="*",
        default=["data/customers_10_vrptw.csv", "data/customers_20_vrptw.csv"],
        help="Customer CSV files to evaluate",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeated runs per instance (default: 5)")
    parser.add_argument("--travel-speed", type=float, default=5.0, help="Travel speed used for time-feasibility")
    parser.add_argument("--rcl-size", type=int, default=1, help="RCL size for randomized NN/CW (1 = deterministic)")
    parser.add_argument("--nn-w-distance", type=float, default=1.0, help="Weight for NN travel-time term")
    parser.add_argument("--nn-w-urgency", type=float, default=0.0, help="Weight for NN urgency/slack term (0 = disabled)")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base seed for reproducible repeated runs (seed = base_seed + run_idx)",
    )
    parser.add_argument(
        "--max-route-duration",
        type=float,
        default=480,
        help="Max route duration in minutes (applied via shared route splitting)",
    )
    parser.add_argument("--max-vehicles", type=int, default=None, help="Optional maximum vehicles")
    parser.add_argument("--no-dual", action="store_true", help="Skip feasible dual-pipeline runs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for customers_file in args.customers:
        instance = VRPTWInstance.load_from_csv(
            customers_file=customers_file,
            distance_file=None,
            depot_x=0.0,
            depot_y=0.0,
            depot_ready=480,
            depot_due=1020,
            service_time=60,
            n_vehicles=args.max_vehicles,
            travel_speed=args.travel_speed,
        )

        records: List[RunRecord] = []

        for run_idx in range(args.repeats):
            seed = args.base_seed + run_idx if args.base_seed is not None else None
            nn_init, nn_imp = _run_initial_and_improved(
                instance,
                "nn",
                args.max_vehicles,
                args.max_route_duration,
                args.rcl_size,
                seed,
                args.nn_w_distance,
                args.nn_w_urgency,
            )
            cw_init, cw_imp = _run_initial_and_improved(
                instance,
                "cw",
                args.max_vehicles,
                args.max_route_duration,
                args.rcl_size,
                seed,
                args.nn_w_distance,
                args.nn_w_urgency,
            )
            records.extend([nn_init, nn_imp, cw_init, cw_imp])

            if not args.no_dual:
                records.append(
                    _run_feasible_dual(
                        instance,
                        args.max_vehicles,
                        args.max_route_duration,
                        args.rcl_size,
                        seed,
                        args.nn_w_distance,
                        args.nn_w_urgency,
                    )
                )

        _print_summary_table(Path(customers_file).name, records)


if __name__ == "__main__":
    main()

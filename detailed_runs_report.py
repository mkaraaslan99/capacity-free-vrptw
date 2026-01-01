#!/usr/bin/env python3
"""Detailed repeated-run report for VRPTW heuristics.

Generates per-run detailed tables + a consolidated CSV so you can compare:
- NN initial
- NN improved (2-opt + relocation)
- CW initial
- CW improved (2-opt + relocation)
- Feasible Dual-Pipeline final

Default experiment setup (matches report request):
- customers_10_vrptw.csv: 5 runs
- customers_20_vrptw.csv: 10 runs

Randomness is controlled via RCL (top-k selection) and seed.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import random

from problems.vrptw import VRPTWInstance
from heuristics.vrptw_nearest_neighbor import nearest_neighbor_vrptw
from heuristics.vrptw_savings_new import clarke_wright_savings_vrptw
from final_feasible_pipeline import final_feasible_dual_pipeline
from local_search.local_search_operators import combined_local_search
from main_capacity_free_vrptw import _apply_max_route_duration


@dataclass(frozen=True)
class Row:
    instance: str
    run_idx: int
    seed: int
    variant: str
    stage: str
    cost: float
    routes: int
    feasible: bool
    time_s: float


def _summ(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "min": min(values), "max": max(values)}


def _print_per_run(instance: str, rows: List[Row]) -> None:
    print("\n" + "=" * 110)
    print(f"DETAILED RUNS | {instance}")
    print("=" * 110)
    header = f"{'run':>3s} | {'seed':>5s} | {'NN_init':>10s} | {'NN_imp':>10s} | {'CW_init':>10s} | {'CW_imp':>10s} | {'Dual_final':>10s}"
    print(header)
    print("-" * len(header))

    # index by (run_idx, variant_stage)
    idx: Dict[Tuple[int, str], float] = {}
    for r in rows:
        key = f"{r.variant}_{r.stage}"
        idx[(r.run_idx, key)] = r.cost

    for run_idx in sorted({r.run_idx for r in rows}):
        seed = next(r.seed for r in rows if r.run_idx == run_idx)
        nn_i = idx.get((run_idx, "NN_initial"), float("nan"))
        nn_p = idx.get((run_idx, "NN_improved"), float("nan"))
        cw_i = idx.get((run_idx, "CW_initial"), float("nan"))
        cw_p = idx.get((run_idx, "CW_improved"), float("nan"))
        df = idx.get((run_idx, "Dual_final"), float("nan"))
        print(f"{run_idx:3d} | {seed:5d} | {nn_i:10.2f} | {nn_p:10.2f} | {cw_i:10.2f} | {cw_p:10.2f} | {df:10.2f}")


def _print_summary(instance: str, rows: List[Row]) -> None:
    print("\n" + "=" * 110)
    print(f"SUMMARY (mean±std) | {instance}")
    print("=" * 110)

    groups: Dict[str, List[Row]] = {}
    for r in rows:
        groups.setdefault(f"{r.variant}_{r.stage}", []).append(r)

    header = f"{'metric':18s} | {'feas%':>6s} | {'cost_mean':>10s} | {'cost_std':>9s} | {'min':>10s} | {'max':>10s} | {'time_mean(s)':>11s}"
    print(header)
    print("-" * len(header))

    for key in ["NN_initial", "NN_improved", "CW_initial", "CW_improved", "Dual_final"]:
        rs = groups.get(key, [])
        costs = [x.cost for x in rs]
        times = [x.time_s for x in rs]
        feas = 100.0 * (sum(1 for x in rs if x.feasible) / len(rs)) if rs else 0.0
        cs = _summ(costs)
        ts = _summ(times)
        print(
            f"{key:18s} | {feas:6.1f} | {cs['mean']:10.2f} | {cs['std']:9.2f} | {cs['min']:10.2f} | {cs['max']:10.2f} | {ts['mean']:11.3f}"
        )


def _run_one(
    instance: VRPTWInstance,
    max_vehicles: Optional[int],
    max_route_duration: Optional[float],
    rcl_size: int,
    seed: int,
    nn_w_distance: float,
    nn_w_urgency: float,
) -> List[Row]:
    rows: List[Row] = []

    # NN initial
    t0 = time.time()
    nn_sol, _, _ = nearest_neighbor_vrptw(
        instance,
        max_vehicles,
        rcl_size=rcl_size,
        seed=seed,
        rng=random.Random(seed),
        w_distance=nn_w_distance,
        w_urgency=nn_w_urgency,
    )
    nn_sol = _apply_max_route_duration(instance, nn_sol, max_route_duration)
    nn_cost = instance.calculate_solution_cost(nn_sol)
    nn_feas = instance.is_solution_feasible(nn_sol)[0]
    rows.append(Row(instance=instance.name, run_idx=-1, seed=seed, variant="NN", stage="initial", cost=nn_cost, routes=len(nn_sol), feasible=nn_feas, time_s=time.time() - t0))

    # NN improved
    t1 = time.time()
    nn_imp_sol, _, _ = combined_local_search(instance, [r.copy() for r in nn_sol])
    nn_imp_sol = _apply_max_route_duration(instance, nn_imp_sol, max_route_duration)
    nn_imp_cost = instance.calculate_solution_cost(nn_imp_sol)
    nn_imp_feas = instance.is_solution_feasible(nn_imp_sol)[0]
    rows.append(Row(instance=instance.name, run_idx=-1, seed=seed, variant="NN", stage="improved", cost=nn_imp_cost, routes=len(nn_imp_sol), feasible=nn_imp_feas, time_s=time.time() - t1))

    # CW initial
    t2 = time.time()
    cw_sol, _, _ = clarke_wright_savings_vrptw(
        instance,
        max_vehicles,
        rcl_size=rcl_size,
        seed=seed,
        rng=random.Random(seed),
    )
    cw_sol = _apply_max_route_duration(instance, cw_sol, max_route_duration)
    cw_cost = instance.calculate_solution_cost(cw_sol)
    cw_feas = instance.is_solution_feasible(cw_sol)[0]
    rows.append(Row(instance=instance.name, run_idx=-1, seed=seed, variant="CW", stage="initial", cost=cw_cost, routes=len(cw_sol), feasible=cw_feas, time_s=time.time() - t2))

    # CW improved
    t3 = time.time()
    cw_imp_sol, _, _ = combined_local_search(instance, [r.copy() for r in cw_sol])
    cw_imp_sol = _apply_max_route_duration(instance, cw_imp_sol, max_route_duration)
    cw_imp_cost = instance.calculate_solution_cost(cw_imp_sol)
    cw_imp_feas = instance.is_solution_feasible(cw_imp_sol)[0]
    rows.append(Row(instance=instance.name, run_idx=-1, seed=seed, variant="CW", stage="improved", cost=cw_imp_cost, routes=len(cw_imp_sol), feasible=cw_imp_feas, time_s=time.time() - t3))

    # Dual final (runs both pipelines and selects best)
    t4 = time.time()
    internal_max_route_duration = 10**9
    with contextlib.redirect_stdout(io.StringIO()):
        dual_sol, _, _ = final_feasible_dual_pipeline(
            instance,
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
    dual_sol = _apply_max_route_duration(instance, dual_sol, max_route_duration)
    dual_cost = instance.calculate_solution_cost(dual_sol)
    dual_feas = instance.is_solution_feasible(dual_sol)[0]
    rows.append(Row(instance=instance.name, run_idx=-1, seed=seed, variant="Dual", stage="final", cost=dual_cost, routes=len(dual_sol), feasible=dual_feas, time_s=time.time() - t4))

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detailed multi-run report for VRPTW NN/CW initial/improved/final")
    p.add_argument("--travel-speed", type=float, default=5.0)
    p.add_argument("--max-route-duration", type=float, default=480)
    p.add_argument("--max-vehicles", type=int, default=None)
    p.add_argument("--rcl-size", type=int, default=3, help="Top-k RCL size (1=deterministic)")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--nn-w-distance", type=float, default=1.0)
    p.add_argument("--nn-w-urgency", type=float, default=0.0)
    p.add_argument(
        "--case",
        action="append",
        default=None,
        help="Case definition as path:repeats (repeatable). Example: --case data/customers_10_vrptw.csv:5",
    )
    p.add_argument("--out", type=str, default="report_outputs/vrptw_detailed_runs.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    plan: List[Tuple[str, int]]
    if args.case:
        plan = []
        for item in args.case:
            if not item or ":" not in item:
                raise ValueError("Each --case must be in the format path:repeats")
            path_str, reps_str = item.rsplit(":", 1)
            reps = int(reps_str)
            if reps < 1:
                raise ValueError("repeats must be >= 1")
            plan.append((path_str, reps))
    else:
        plan = [
            ("data/customers_10_vrptw.csv", 5),
            ("data/customers_20_vrptw.csv", 10),
        ]

    all_rows: List[Row] = []

    for customers_file, repeats in plan:
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

        inst_rows: List[Row] = []
        for run_idx in range(repeats):
            seed = args.base_seed + run_idx
            rows = _run_one(
                instance,
                args.max_vehicles,
                args.max_route_duration,
                args.rcl_size,
                seed,
                args.nn_w_distance,
                args.nn_w_urgency,
            )
            # patch run_idx into rows
            for r in rows:
                inst_rows.append(
                    Row(
                        instance=r.instance,
                        run_idx=run_idx,
                        seed=r.seed,
                        variant=r.variant,
                        stage=r.stage,
                        cost=r.cost,
                        routes=r.routes,
                        feasible=r.feasible,
                        time_s=r.time_s,
                    )
                )

        _print_per_run(Path(customers_file).name, inst_rows)
        _print_summary(Path(customers_file).name, inst_rows)

        all_rows.extend(inst_rows)

    out_path = Path(args.out)
    # csv writer needs one-pass iterable; all_rows is list so OK
    if not all_rows:
        raise RuntimeError("No rows generated")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))

    print(f"\nCSV written to: {out_path}")


if __name__ == "__main__":
    main()

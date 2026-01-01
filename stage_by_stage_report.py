import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd

from problems.vrptw import VRPTWInstance
from heuristics.vrptw_nearest_neighbor import nearest_neighbor_vrptw
from heuristics.vrptw_savings_new import clarke_wright_savings_vrptw
from local_search.local_search_operators import two_opt_local_search, relocation_local_search
from utils.visualization_new import plot_vrptw_solution, print_solution_summary


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_and_record_stage(
    vrptw: VRPTWInstance,
    dataset_label: str,
    method: str,
    stage: str,
    solution,
    cost: float,
    stage_time: float,
    out_dir: Path,
    records: list,
) -> None:
    feasible, reason = vrptw.is_solution_feasible(solution)

    records.append({
        "dataset": dataset_label,
        "method": method,
        "stage": stage,
        "cost": cost,
        "routes": len(solution),
        "feasible": bool(feasible),
        "feasibility_reason": reason,
        "time_seconds": stage_time,
    })

    plot_path = out_dir / f"{method.lower()}_{stage.lower()}_solution.png"
    plot_vrptw_solution(
        vrptw,
        solution,
        title=f"{dataset_label} | {method} | {stage}",
        save_path=str(plot_path),
    )


def run_stage_by_stage(customers_file: str, travel_speed: float, out_root: Path, max_vehicles: int | None) -> None:
    vrptw = VRPTWInstance.load_from_csv(customers_file=customers_file, travel_speed=travel_speed, n_vehicles=max_vehicles)
    dataset_label = Path(customers_file).stem

    out_dir = out_root / dataset_label
    _ensure_dir(out_dir)

    records: list[dict] = []

    print("=" * 70)
    print(f"DATASET: {customers_file}")
    print(f"Travel speed (constant): {travel_speed}")
    print("=" * 70)

    print("\n" + "=" * 60)
    print("10/20 STYLE OUTPUT (NN THEN CW)" if "customers_" in dataset_label else "")
    print("=" * 60)

    # 1) Nearest Neighbor pipeline
    print("\n" + "#" * 60)
    print("NEAREST NEIGHBOR PIPELINE")
    print("#" * 60)

    t0 = time.time()
    nn_solution, nn_cost, _ = nearest_neighbor_vrptw(vrptw, max_vehicles=max_vehicles)
    nn_t = time.time() - t0
    print_solution_summary(vrptw, nn_solution, nn_cost, "NN Initial")
    _run_and_record_stage(vrptw, dataset_label, "NN", "Initial", deepcopy(nn_solution), nn_cost, nn_t, out_dir, records)

    t0 = time.time()
    nn_after_2opt, nn_cost_2opt, _ = two_opt_local_search(vrptw, deepcopy(nn_solution))
    t_2opt = time.time() - t0
    print_solution_summary(vrptw, nn_after_2opt, nn_cost_2opt, "NN After 2-opt")
    print(f"Improvement (distance) after 2-opt: {nn_cost - nn_cost_2opt:.2f}")
    _run_and_record_stage(vrptw, dataset_label, "NN", "After_2opt", deepcopy(nn_after_2opt), nn_cost_2opt, t_2opt, out_dir, records)

    t0 = time.time()
    nn_after_reloc, nn_cost_reloc, _ = relocation_local_search(vrptw, deepcopy(nn_after_2opt))
    t_reloc = time.time() - t0
    print_solution_summary(vrptw, nn_after_reloc, nn_cost_reloc, "NN After Relocation")
    print(f"Improvement (distance) after relocation: {nn_cost_2opt - nn_cost_reloc:.2f}")
    print(f"Total improvement (distance) NN: {nn_cost - nn_cost_reloc:.2f}")
    _run_and_record_stage(vrptw, dataset_label, "NN", "After_relocation", deepcopy(nn_after_reloc), nn_cost_reloc, t_reloc, out_dir, records)

    # 2) Clarke-Wright pipeline
    print("\n" + "#" * 60)
    print("CLARKE-WRIGHT PIPELINE")
    print("#" * 60)

    t0 = time.time()
    cw_solution, cw_cost, _ = clarke_wright_savings_vrptw(vrptw, max_vehicles=max_vehicles)
    cw_t = time.time() - t0
    print_solution_summary(vrptw, cw_solution, cw_cost, "CW Initial")
    _run_and_record_stage(vrptw, dataset_label, "CW", "Initial", deepcopy(cw_solution), cw_cost, cw_t, out_dir, records)

    t0 = time.time()
    cw_after_2opt, cw_cost_2opt, _ = two_opt_local_search(vrptw, deepcopy(cw_solution))
    t_2opt = time.time() - t0
    print_solution_summary(vrptw, cw_after_2opt, cw_cost_2opt, "CW After 2-opt")
    print(f"Improvement (distance) after 2-opt: {cw_cost - cw_cost_2opt:.2f}")
    _run_and_record_stage(vrptw, dataset_label, "CW", "After_2opt", deepcopy(cw_after_2opt), cw_cost_2opt, t_2opt, out_dir, records)

    t0 = time.time()
    cw_after_reloc, cw_cost_reloc, _ = relocation_local_search(vrptw, deepcopy(cw_after_2opt))
    t_reloc = time.time() - t0
    print_solution_summary(vrptw, cw_after_reloc, cw_cost_reloc, "CW After Relocation")
    print(f"Improvement (distance) after relocation: {cw_cost_2opt - cw_cost_reloc:.2f}")
    print(f"Total improvement (distance) CW: {cw_cost - cw_cost_reloc:.2f}")
    _run_and_record_stage(vrptw, dataset_label, "CW", "After_relocation", deepcopy(cw_after_reloc), cw_cost_reloc, t_reloc, out_dir, records)

    df = pd.DataFrame(records)
    csv_path = out_dir / "stage_by_stage_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved stage-by-stage table to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default="results/stage_by_stage")
    parser.add_argument("--max-vehicles", type=int, default=0)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    _ensure_dir(out_root)

    max_vehicles = None if args.max_vehicles == 0 else args.max_vehicles

    run_stage_by_stage("data/customers_10_vrptw.csv", args.speed, out_root, max_vehicles)
    run_stage_by_stage("data/customers_20_vrptw.csv", args.speed, out_root, max_vehicles)


if __name__ == "__main__":
    main()

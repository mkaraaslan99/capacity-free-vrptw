import os
os.environ.setdefault("MPLBACKEND", "Agg")

import time
from pathlib import Path

import pandas as pd

from problems.vrptw import VRPTWInstance
from heuristics.vrptw_nearest_neighbor import nearest_neighbor_vrptw
from heuristics.vrptw_savings_new import clarke_wright_savings_vrptw
from final_feasible_pipeline import final_feasible_dual_pipeline
from main_capacity_free_vrptw import _apply_max_route_duration


def _run_one(vrptw: VRPTWInstance, algorithm: str, max_route_duration: float):
    t0 = time.perf_counter()

    if algorithm == "Nearest Neighbor":
        solution, _, _ = nearest_neighbor_vrptw(vrptw, max_vehicles=None)
    elif algorithm == "Clarke-Wright":
        solution, _, _ = clarke_wright_savings_vrptw(vrptw, max_vehicles=None)
    elif algorithm == "Feasible Dual-Pipeline":
        # Disable internal splitting; we apply shared splitting uniformly below.
        solution, _, _ = final_feasible_dual_pipeline(
            vrptw,
            max_vehicles=None,
            nn_variant="sequential",
            cw_variant="standard",
            max_route_duration=10**9,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    solution = _apply_max_route_duration(vrptw, solution, max_route_duration)

    cost = vrptw.calculate_solution_cost(solution)
    routes = len(solution)
    feasible, _ = vrptw.is_solution_feasible(solution)

    elapsed = time.perf_counter() - t0

    return {
        "cost": float(cost),
        "routes": int(routes),
        "feasible": bool(feasible),
        "time_seconds": float(elapsed),
    }


def main():
    travel_speed = 5.0
    max_durations = [480, 540, 600]
    datasets = [
        "data/customers_10_vrptw.csv",
        "data/customers_20_vrptw.csv",
    ]
    algorithms = [
        "Nearest Neighbor",
        "Clarke-Wright",
        "Feasible Dual-Pipeline",
    ]

    out_dir = Path("results/duration_sweep_speed5")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for customers_file in datasets:
        vrptw = VRPTWInstance.load_from_csv(
            customers_file=customers_file,
            travel_speed=travel_speed,
            n_vehicles=None,
        )
        dataset_label = Path(customers_file).stem

        for d in max_durations:
            for alg in algorithms:
                res = _run_one(vrptw, alg, d)
                records.append(
                    {
                        "dataset": dataset_label,
                        "travel_speed": travel_speed,
                        "max_route_duration": d,
                        "algorithm": alg,
                        **res,
                    }
                )

    df = pd.DataFrame(records)
    
    # Save CSV files
    csv_path = out_dir / "duration_sweep.csv"
    df.to_csv(csv_path, index=False)

    # Also save a pivot for easier report inclusion
    pivot = df.pivot_table(
        index=["dataset", "max_route_duration"],
        columns="algorithm",
        values="cost",
        aggfunc="first",
    ).reset_index()
    pivot_path = out_dir / "duration_sweep_cost_pivot.csv"
    pivot.to_csv(pivot_path, index=False)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved CSV: {pivot_path}")
    
    # Save Excel files
    excel_path = out_dir / "duration_sweep.xlsx"
    df.to_excel(excel_path, index=False, sheet_name='Duration_Sweep')
    
    pivot_excel_path = out_dir / "duration_sweep_cost_pivot.xlsx"
    pivot.to_excel(pivot_excel_path, index=False, sheet_name='Cost_Pivot')
    
    # Combined Excel with multiple sheets
    excel_combined = out_dir / "duration_sweep_all.xlsx"
    with pd.ExcelWriter(excel_combined, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Raw_Data', index=False)
        pivot.to_excel(writer, sheet_name='Cost_Pivot', index=False)
    
    print(f"Saved Excel: {excel_path}")
    print(f"Saved Excel: {pivot_excel_path}")
    print(f"Saved Excel: {excel_combined} (combined)")


if __name__ == "__main__":
    main()

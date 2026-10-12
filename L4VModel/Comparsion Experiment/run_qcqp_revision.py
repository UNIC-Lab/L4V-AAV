"""Run and persist the revision-round SCA-QCQP comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import cvxpy as cp
import numpy as np

from qcqp_optimization import qcqp_optimize


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results_round2" / "qcqp"


def run(trials: int, sca_iters: int, smoke: bool):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(40926000, 40926000 + trials, dtype=np.int64)
    rows = []
    paths = []
    users_all = []
    tasks_all = []
    for index, seed in enumerate(seeds):
        started = time.perf_counter()
        result, users, tasks = qcqp_optimize(
            seed=int(seed), max_horizon=80 if smoke else 160,
            sca_iters=sca_iters,
        )
        row = {
            "trial": index,
            "seed": int(seed),
            "mission_time_slots": result.mission_time,
            "average_completion_time_slots": result.average_completion_time,
            "solve_time_s": result.solve_time_s,
            "horizon": result.horizon,
            "sca_iterations": result.sca_iterations,
            "status": result.status,
            "max_service_deficit": result.max_service_deficit,
            "wall_time_s": time.perf_counter() - started,
        }
        rows.append(row)
        paths.append(result.path)
        users_all.append(users)
        tasks_all.append(tasks)
        print(json.dumps(row), flush=True)

    npz_path = OUTPUT / ("qcqp_smoke.npz" if smoke else "qcqp_formal.npz")
    np.savez_compressed(
        npz_path,
        seeds=seeds,
        mission_time_slots=np.asarray([r["mission_time_slots"] for r in rows]),
        average_completion_time_slots=np.asarray([r["average_completion_time_slots"] for r in rows]),
        solve_time_s=np.asarray([r["solve_time_s"] for r in rows]),
        horizons=np.asarray([r["horizon"] for r in rows]),
        sca_iterations=np.asarray([r["sca_iterations"] for r in rows]),
        max_service_deficit=np.asarray([r["max_service_deficit"] for r in rows]),
        paths=np.asarray(paths, dtype=object),
        user_positions=np.asarray(users_all),
        user_tasks=np.asarray(tasks_all),
    )
    summary = {
        "protocol": "round2_sca_qcqp_smoke" if smoke else "round2_sca_qcqp_formal",
        "trials": trials,
        "seeds": seeds.tolist(),
        "solver": "CLARABEL with SCS fallback",
        "cvxpy": cp.__version__,
        "python": platform.python_version(),
        "median_mission_time_slots": float(np.median([r["mission_time_slots"] for r in rows])),
        "median_average_completion_time_slots": float(np.median([r["average_completion_time_slots"] for r in rows])),
        "median_solve_time_s": float(np.median([r["solve_time_s"] for r in rows])),
        "maximum_service_deficit": float(np.max([r["max_service_deficit"] for r in rows])),
        "raw_npz": str(npz_path.relative_to(ROOT)),
    }
    summary_path = OUTPUT / ("summary_smoke.json" if smoke else "summary_formal.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--sca-iters", type=int, default=6)
    args = parser.parse_args()
    run(args.trials or (1 if args.smoke else 10), 2 if args.smoke else args.sca_iters, args.smoke)

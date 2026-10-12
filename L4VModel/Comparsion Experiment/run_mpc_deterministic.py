"""
run_mpc_deterministic.py

Adds MPC to the ORIGINAL, deterministic-channel comparison (area_size, sigma,
num_users sweeps) so that Figs. 2-4 have the same six-method roster
(L4V, GA, DQN, A2C, DDPG, MPC) as the rest of the paper, instead of MPC only
appearing in one figure. Uses shadow_std=0, rician_k=None, use_allocation=False
so MPC faces the identical deterministic, equal-split channel that GA, DQN,
A2C, and DDPG were originally evaluated on (see results/data/*.csv, already on
disk from the original experiment run), and the same n_repeat=10 and default
parameters as run_dso.py / run_ga.py / etc., so the new MPC rows are directly
comparable to the existing data without rerunning the other five methods.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from mpc_optimization_extended import mpc_optimize

RESULTS_DIR = "results/data"
os.makedirs(RESULTS_DIR, exist_ok=True)

FIXED = dict(h_unit=1, area_size=10, num_users=4, max_step=0.2, time_step_duration=0.1, sigma=0.1)

SWEEPS = {
    "area_size": [5, 10, 15, 20, 25],
    "sigma": [0.05, 0.1, 0.15, 0.2, 0.25],
    "num_users": [2, 4, 6, 8, 10],
    "h_unit": [0.5, 1.0, 1.5, 2.0, 2.5],
}

N_REPEAT = 10


def run_one(param_name, value, n_repeat=N_REPEAT, tag=""):
    print(f"=== MPC deterministic sweep: {param_name} = {value} ===")
    records = []
    for i in range(n_repeat):
        params = dict(FIXED)
        params[param_name] = value
        seed = int(value * 1000) + 7000 + i
        t0 = time.time()
        result = mpc_optimize(
            num_users=params["num_users"], area_size=params["area_size"],
            h_unit=params["h_unit"], sigma=params["sigma"],
            max_step=params["max_step"], time_step_duration=params["time_step_duration"],
            horizon=15, inner_iters=25, max_episodes=500,
            use_allocation=False, shadow_std=0.0, rician_k=None, seed=seed,
        )
        wall = time.time() - t0
        rec = {
            "run_index": i + 1,
            "time_steps": result.get("time_steps", np.nan),
            "avg_task_reduction": result.get("avg_task_reduction", np.nan),
            "convergence_episode": result.get("convergence_episode", np.nan),
            "total_training_time": result.get("total_training_time", np.nan),
        }
        rates = np.array(result.get("transmission_rates", []), dtype=float)
        rec["avg_transmission_rate"] = float(np.mean(rates)) if len(rates) > 0 else np.nan
        records.append(rec)
        print(f"  run {i+1}/{n_repeat}: time_steps={rec['time_steps']} wall={wall:.1f}s")

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, f"mpc_result_{param_name}_{value}{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", choices=list(SWEEPS.keys()), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        run_one(args.param, SWEEPS[args.param][0], n_repeat=1, tag="_smoke")
        print("Smoke test PASSED")
    else:
        for v in SWEEPS[args.param]:
            run_one(args.param, v)

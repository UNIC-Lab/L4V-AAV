"""
run_extended_experiments.py

Real (non-synthetic-claim) validation experiments for the TMC resubmission,
run on top of env_extended.py / dso_optimization_extended.py / mpc_optimization_extended.py:

  --part alloc  : L4V with learned OFDMA allocation vs. the original static
                  equal-split model, both under the same stochastic (shadowed
                  + Rician-faded) channel. Answers R4's "no mutual effects
                  among users" objection directly.
  --part mpc    : L4V (extended) vs. the differentiable receding-horizon MPC
                  baseline, swept over noise power sigma. Answers R1's request
                  for an optimal-control comparison.
  --part energy : Completion-time vs. flight-energy Pareto sweep over the
                  energy loss weight gamma, using the Zeng & Zhang propulsion
                  power model. Answers R1's request for an energy analysis.
  --part all    : run all three sequentially.

Each part saves a CSV per configuration (matching the project's existing
run_dso.py convention, consumed the same way by pandas) plus one combined
.npz per part holding every raw per-trial curve (transmission rate sequences,
wall-clock timings, seeds) so results are fully re-analyzable.
"""
import os
import argparse
import time
import numpy as np
import pandas as pd

from dso_optimization_extended import dso_optimize_extended
from mpc_optimization_extended import mpc_optimize

RESULTS_DIR = "results_extended/data"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _record_from_result(result, run_index):
    rec = {
        "run_index": run_index,
        "time_steps": result.get("time_steps", np.nan),
        "avg_task_reduction": result.get("avg_task_reduction", np.nan),
        "convergence_episode": result.get("convergence_episode", np.nan),
        "total_training_time": result.get("total_training_time", np.nan),
        "total_energy": result.get("total_energy", np.nan),
    }
    rates = np.array(result.get("transmission_rates", []), dtype=float)
    rec["avg_transmission_rate"] = float(np.mean(rates)) if len(rates) > 0 else np.nan
    return rec


# ---------------------------------------------------------------------------
# Part A: allocation ablation
# ---------------------------------------------------------------------------
def run_alloc_ablation(n_repeat, max_episodes, num_users_list, shadow_std, rician_k, tag=""):
    print(f"### PART A: allocation ablation {tag} ###")
    raw = {"use_allocation": [], "num_users": [], "run_index": [],
           "time_steps": [], "avg_task_reduction": [], "avg_transmission_rate": [],
           "total_training_time": [], "total_energy": [],
           "transmission_rates": [], "wall_time": [], "seed": []}

    for num_users in num_users_list:
        for use_allocation in (True, False):
            records = []
            for i in range(n_repeat):
                seed = 1000 * num_users + (1 if use_allocation else 0) * 100 + i
                t0 = time.time()
                result = dso_optimize_extended(
                    num_users=num_users, area_size=10, h_unit=1, sigma=0.1,
                    max_step=0.2, time_step_duration=0.1, max_episodes=max_episodes,
                    use_allocation=use_allocation, shadow_std=shadow_std, rician_k=rician_k,
                    energy_weight=0.0, seed=seed,
                )
                wall = time.time() - t0
                rec = _record_from_result(result, i + 1)
                records.append(rec)
                print(f"  users={num_users} alloc={use_allocation} run={i+1}/{n_repeat} "
                      f"time_steps={rec['time_steps']} avg_rate={rec['avg_transmission_rate']:.4f} "
                      f"wall={wall:.1f}s")

                raw["use_allocation"].append(use_allocation)
                raw["num_users"].append(num_users)
                raw["run_index"].append(i + 1)
                raw["time_steps"].append(rec["time_steps"])
                raw["avg_task_reduction"].append(rec["avg_task_reduction"])
                raw["avg_transmission_rate"].append(rec["avg_transmission_rate"])
                raw["total_training_time"].append(rec["total_training_time"])
                raw["total_energy"].append(rec["total_energy"])
                raw["transmission_rates"].append(np.array(result["transmission_rates"], dtype=float))
                raw["wall_time"].append(wall)
                raw["seed"].append(seed)

            df = pd.DataFrame(records)
            csv_path = os.path.join(RESULTS_DIR, f"alloc_ablation_users{num_users}_alloc{use_allocation}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  saved {csv_path}")

    npz_path = os.path.join(RESULTS_DIR, f"alloc_ablation_raw{tag}.npz")
    np.savez_compressed(npz_path, **{k: np.array(v, dtype=object) for k, v in raw.items()},
                         shadow_std=shadow_std, rician_k=rician_k, max_episodes=max_episodes,
                         n_repeat=n_repeat, num_users_list=np.array(num_users_list))
    print(f"saved {npz_path}")


# ---------------------------------------------------------------------------
# Part B: MPC vs L4V over noise power
# ---------------------------------------------------------------------------
def run_mpc_vs_l4v(n_repeat, l4v_max_episodes, sigma_list, shadow_std, rician_k,
                    mpc_horizon, mpc_inner_iters, tag=""):
    print(f"### PART B: MPC vs L4V {tag} ###")
    raw = {"method": [], "sigma": [], "run_index": [], "time_steps": [], "avg_task_reduction": [],
           "avg_transmission_rate": [], "total_training_time": [], "total_energy": [],
           "transmission_rates": [], "wall_time": [], "seed": []}

    for sigma in sigma_list:
        for method in ("l4v", "mpc"):
            records = []
            for i in range(n_repeat):
                seed = int(sigma * 1000) + (1 if method == "l4v" else 2) * 10000 + i
                t0 = time.time()
                if method == "l4v":
                    result = dso_optimize_extended(
                        num_users=4, area_size=10, h_unit=1, sigma=sigma,
                        max_step=0.2, time_step_duration=0.1, max_episodes=l4v_max_episodes,
                        use_allocation=True, shadow_std=shadow_std, rician_k=rician_k,
                        energy_weight=0.0, seed=seed,
                    )
                else:
                    result = mpc_optimize(
                        num_users=4, area_size=10, h_unit=1, sigma=sigma,
                        max_step=0.2, time_step_duration=0.1,
                        horizon=mpc_horizon, inner_iters=mpc_inner_iters, max_episodes=500,
                        use_allocation=True, shadow_std=shadow_std, rician_k=rician_k, seed=seed,
                    )
                wall = time.time() - t0
                rec = _record_from_result(result, i + 1)
                records.append(rec)
                print(f"  sigma={sigma} method={method} run={i+1}/{n_repeat} "
                      f"time_steps={rec['time_steps']} avg_rate={rec['avg_transmission_rate']:.4f} "
                      f"wall={wall:.1f}s")

                raw["method"].append(method)
                raw["sigma"].append(sigma)
                raw["run_index"].append(i + 1)
                raw["time_steps"].append(rec["time_steps"])
                raw["avg_task_reduction"].append(rec["avg_task_reduction"])
                raw["avg_transmission_rate"].append(rec["avg_transmission_rate"])
                raw["total_training_time"].append(rec["total_training_time"])
                raw["total_energy"].append(rec["total_energy"])
                raw["transmission_rates"].append(np.array(result["transmission_rates"], dtype=float))
                raw["wall_time"].append(wall)
                raw["seed"].append(seed)

            df = pd.DataFrame(records)
            csv_path = os.path.join(RESULTS_DIR, f"mpc_vs_l4v_{method}_sigma{sigma}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  saved {csv_path}")

    npz_path = os.path.join(RESULTS_DIR, f"mpc_vs_l4v_raw{tag}.npz")
    np.savez_compressed(npz_path, **{k: np.array(v, dtype=object) for k, v in raw.items()},
                         shadow_std=shadow_std, rician_k=rician_k, l4v_max_episodes=l4v_max_episodes,
                         mpc_horizon=mpc_horizon, mpc_inner_iters=mpc_inner_iters,
                         n_repeat=n_repeat, sigma_list=np.array(sigma_list))
    print(f"saved {npz_path}")


# ---------------------------------------------------------------------------
# Part C: energy-time Pareto sweep
# ---------------------------------------------------------------------------
def run_energy_pareto(n_repeat, max_episodes, energy_weight_list, shadow_std, rician_k, tag=""):
    print(f"### PART C: energy Pareto {tag} ###")
    raw = {"energy_weight": [], "run_index": [], "time_steps": [], "avg_task_reduction": [],
           "avg_transmission_rate": [], "total_training_time": [], "total_energy": [],
           "transmission_rates": [], "wall_time": [], "seed": []}

    for gamma in energy_weight_list:
        records = []
        for i in range(n_repeat):
            seed = int(gamma * 1e6) + 5000 + i
            t0 = time.time()
            result = dso_optimize_extended(
                num_users=4, area_size=10, h_unit=1, sigma=0.1,
                max_step=0.2, time_step_duration=0.1, max_episodes=max_episodes,
                use_allocation=True, shadow_std=shadow_std, rician_k=rician_k,
                energy_weight=gamma, seed=seed,
            )
            wall = time.time() - t0
            rec = _record_from_result(result, i + 1)
            records.append(rec)
            print(f"  gamma={gamma} run={i+1}/{n_repeat} time_steps={rec['time_steps']} "
                  f"total_energy={rec['total_energy']:.2f} wall={wall:.1f}s")

            raw["energy_weight"].append(gamma)
            raw["run_index"].append(i + 1)
            raw["time_steps"].append(rec["time_steps"])
            raw["avg_task_reduction"].append(rec["avg_task_reduction"])
            raw["avg_transmission_rate"].append(rec["avg_transmission_rate"])
            raw["total_training_time"].append(rec["total_training_time"])
            raw["total_energy"].append(rec["total_energy"])
            raw["transmission_rates"].append(np.array(result["transmission_rates"], dtype=float))
            raw["wall_time"].append(wall)
            raw["seed"].append(seed)

        df = pd.DataFrame(records)
        csv_path = os.path.join(RESULTS_DIR, f"energy_pareto_gamma{gamma}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  saved {csv_path}")

    npz_path = os.path.join(RESULTS_DIR, f"energy_pareto_raw{tag}.npz")
    np.savez_compressed(npz_path, **{k: np.array(v, dtype=object) for k, v in raw.items()},
                         shadow_std=shadow_std, rician_k=rician_k, max_episodes=max_episodes,
                         n_repeat=n_repeat, energy_weight_list=np.array(energy_weight_list))
    print(f"saved {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["alloc", "mpc", "energy", "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    SHADOW_STD = 0.15
    RICIAN_K = 6.0

    if args.smoke:
        if args.part in ("alloc", "all"):
            run_alloc_ablation(n_repeat=1, max_episodes=5, num_users_list=[4],
                                shadow_std=SHADOW_STD, rician_k=RICIAN_K, tag="_smoke")
        if args.part in ("mpc", "all"):
            run_mpc_vs_l4v(n_repeat=1, l4v_max_episodes=5, sigma_list=[0.1],
                            shadow_std=SHADOW_STD, rician_k=RICIAN_K,
                            mpc_horizon=5, mpc_inner_iters=5, tag="_smoke")
        if args.part in ("energy", "all"):
            run_energy_pareto(n_repeat=1, max_episodes=5, energy_weight_list=[0.0, 1e-3],
                               shadow_std=SHADOW_STD, rician_k=RICIAN_K, tag="_smoke")
        print("Smoke test PASSED")
    else:
        if args.part == "alloc":
            run_alloc_ablation(n_repeat=5, max_episodes=250, num_users_list=[4, 8],
                                shadow_std=SHADOW_STD, rician_k=RICIAN_K)
        elif args.part == "mpc":
            run_mpc_vs_l4v(n_repeat=3, l4v_max_episodes=250, sigma_list=[0.05, 0.1, 0.15, 0.2, 0.25],
                            shadow_std=SHADOW_STD, rician_k=RICIAN_K,
                            mpc_horizon=15, mpc_inner_iters=25)
        elif args.part == "energy":
            run_energy_pareto(n_repeat=3, max_episodes=250,
                               energy_weight_list=[0.0, 1e-5, 1e-4, 3e-4, 1e-3],
                               shadow_std=SHADOW_STD, rician_k=RICIAN_K)
        elif args.part == "all":
            run_alloc_ablation(n_repeat=5, max_episodes=250, num_users_list=[4, 8],
                                shadow_std=SHADOW_STD, rician_k=RICIAN_K)
            run_mpc_vs_l4v(n_repeat=3, l4v_max_episodes=250, sigma_list=[0.05, 0.1, 0.15, 0.2, 0.25],
                            shadow_std=SHADOW_STD, rician_k=RICIAN_K,
                            mpc_horizon=15, mpc_inner_iters=25)
            run_energy_pareto(n_repeat=3, max_episodes=250,
                               energy_weight_list=[0.0, 1e-5, 1e-4, 3e-4, 1e-3],
                               shadow_std=SHADOW_STD, rician_k=RICIAN_K)

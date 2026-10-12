"""
run_energy_full.py

Full-budget, paired-instance completion-time-versus-energy sweep, replacing the
earlier validation-scale (250-episode, n=3) version whose non-converged seeds
forced defensive exclusion. For each trial index i we fix a seed shared across
ALL energy weights gamma, so every gamma is trained on the identical instance
(same user layout, same task demands, same initial weights); the resulting
time/energy differences isolate the effect of gamma rather than instance noise.
All arms run under the reparameterized shadowed-and-faded channel (kappa=0.15,
K=6) with the learned allocation head, matching run_alloc_full.py.

Full budget: max_episodes=2000, n=5 paired trials, gamma sweep.
"""
import os
import time
import argparse
import numpy as np
import pandas as pd

from dso_optimization_extended import dso_optimize_extended

RESULTS_DIR = "results_extended/data"
os.makedirs(RESULTS_DIR, exist_ok=True)

SHADOW_STD = 0.15
RICIAN_K = 6.0


def run(gamma_list, n_repeat, max_episodes, v_scale, tag=""):
    print(f"### Full-budget paired energy sweep {tag} ###")
    raw = {"energy_weight": [], "run_index": [], "seed": [], "time_steps": [],
           "avg_task_reduction": [], "avg_transmission_rate": [], "total_training_time": [],
           "convergence_episode": [], "total_energy": [], "transmission_rates": []}

    for gamma in gamma_list:
        records = []
        for i in range(n_repeat):
            # Seed depends only on trial index, NOT gamma => paired across gammas.
            seed = 30240000 + i
            t0 = time.time()
            result = dso_optimize_extended(
                num_users=4, area_size=10, h_unit=1, sigma=0.1,
                max_step=0.2, time_step_duration=0.1, max_episodes=max_episodes,
                use_allocation=True, shadow_std=SHADOW_STD, rician_k=RICIAN_K,
                energy_weight=gamma, v_scale=v_scale, seed=seed,
            )
            wall = time.time() - t0
            rec = {
                "run_index": i + 1,
                "time_steps": result.get("time_steps", np.nan),
                "avg_task_reduction": result.get("avg_task_reduction", np.nan),
                "convergence_episode": result.get("convergence_episode", np.nan),
                "total_training_time": result.get("total_training_time", np.nan),
                "total_energy": result.get("total_energy", np.nan),
            }
            rates = np.array(result.get("transmission_rates", []), dtype=float)
            rec["avg_transmission_rate"] = float(np.mean(rates)) if len(rates) > 0 else np.nan
            records.append(rec)
            print(f"  gamma={gamma:g} run={i+1}/{n_repeat} steps={rec['time_steps']} "
                  f"energy={rec['total_energy']:.0f} goodput={rec['avg_task_reduction']:.3f} "
                  f"wall={wall:.0f}s")

            raw["energy_weight"].append(gamma)
            raw["run_index"].append(i + 1)
            raw["seed"].append(seed)
            for k in ("time_steps", "avg_task_reduction", "avg_transmission_rate",
                      "total_training_time", "convergence_episode", "total_energy"):
                raw[k].append(rec[k])
            raw["transmission_rates"].append(np.array(result["transmission_rates"], dtype=float))

        df = pd.DataFrame(records)
        csv = os.path.join(RESULTS_DIR, f"energyfull_gamma{gamma}{tag}.csv")
        df.to_csv(csv, index=False)
        print(f"  saved {csv}")

    npz = os.path.join(RESULTS_DIR, f"energyfull_raw{tag}.npz")
    np.savez_compressed(npz, **{k: np.array(v, dtype=object) for k, v in raw.items()},
                        shadow_std=SHADOW_STD, rician_k=RICIAN_K, max_episodes=max_episodes,
                        n_repeat=n_repeat, v_scale=v_scale, gamma_list=np.array(gamma_list))
    print(f"saved {npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    # v_scale chosen so energy magnitude (~10^3) x gamma (~10^-4) is commensurate
    # with the normalized backlog cost (~O(1)); matches env_extended default.
    if args.smoke:
        run(gamma_list=[0.0, 1e-4], n_repeat=1, max_episodes=8, v_scale=1.0, tag="_smoke")
        print("Smoke test PASSED")
    else:
        run(gamma_list=[0.0, 1e-5, 1e-4, 3e-4, 1e-3], n_repeat=5, max_episodes=2000, v_scale=1.0)

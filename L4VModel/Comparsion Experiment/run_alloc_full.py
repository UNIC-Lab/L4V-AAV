"""
run_alloc_full.py

Full-budget, paired-instance allocation ablation for the TMC resubmission,
replacing the earlier validation-scale (250-episode, n=3) version. For each
trial we draw ONE random user layout / task demand set via a fixed seed, then
train two policies on that identical instance: one with the learned softmax
bandwidth allocation head active, one pinned to the original equal split
(a_i = 1/N). Because both arms share the same seed, they see the same initial
environment and the same initial network weights, so the completion-time /
energy difference isolates the effect of learning the allocation rather than
instance-to-instance variation. Both arms run under the identical
reparameterized shadowed-and-faded channel (kappa=0.15, K=6), so this single
experiment answers both the "no inter-user coupling" (OFDMA) and the
"stochastic channel" reviewer concerns at once.

Full budget: max_episodes=2000 (the dso default), n=5 paired trials, K in {4,8}.
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


def run(num_users_list, n_repeat, max_episodes, tag=""):
    print(f"### Full-budget paired allocation ablation {tag} ###")
    raw = {"use_allocation": [], "num_users": [], "run_index": [], "seed": [],
           "time_steps": [], "avg_task_reduction": [], "avg_transmission_rate": [],
           "total_training_time": [], "convergence_episode": [], "total_energy": [],
           "transmission_rates": []}

    for num_users in num_users_list:
        for use_allocation in (True, False):
            records = []
            for i in range(n_repeat):
                # SAME seed across the two arms => same paired instance.
                seed = 20240000 + num_users * 100 + i
                t0 = time.time()
                result = dso_optimize_extended(
                    num_users=num_users, area_size=10, h_unit=1, sigma=0.1,
                    max_step=0.2, time_step_duration=0.1, max_episodes=max_episodes,
                    use_allocation=use_allocation, shadow_std=SHADOW_STD, rician_k=RICIAN_K,
                    energy_weight=0.0, seed=seed,
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
                print(f"  users={num_users} alloc={use_allocation} run={i+1}/{n_repeat} "
                      f"steps={rec['time_steps']} goodput={rec['avg_task_reduction']:.3f} "
                      f"energy={rec['total_energy']:.0f} wall={wall:.0f}s")

                raw["use_allocation"].append(use_allocation)
                raw["num_users"].append(num_users)
                raw["run_index"].append(i + 1)
                raw["seed"].append(seed)
                for k in ("time_steps", "avg_task_reduction", "avg_transmission_rate",
                          "total_training_time", "convergence_episode", "total_energy"):
                    raw[k].append(rec[k])
                raw["transmission_rates"].append(np.array(result["transmission_rates"], dtype=float))

            df = pd.DataFrame(records)
            csv = os.path.join(RESULTS_DIR, f"allocfull_users{num_users}_alloc{use_allocation}{tag}.csv")
            df.to_csv(csv, index=False)
            print(f"  saved {csv}")

    npz = os.path.join(RESULTS_DIR, f"allocfull_raw{tag}.npz")
    np.savez_compressed(npz, **{k: np.array(v, dtype=object) for k, v in raw.items()},
                        shadow_std=SHADOW_STD, rician_k=RICIAN_K, max_episodes=max_episodes,
                        n_repeat=n_repeat, num_users_list=np.array(num_users_list))
    print(f"saved {npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run(num_users_list=[4], n_repeat=1, max_episodes=8, tag="_smoke")
        print("Smoke test PASSED")
    else:
        run(num_users_list=[4, 8], n_repeat=5, max_episodes=2000)

"""Paired high-speed energy experiment for the second TMC revision.

One normalized distance unit is mapped to 10 m and each control slot lasts
0.1 s.  The conversion factor is therefore 100 s^-1, so the normalized
maximum displacement 0.2 corresponds to 20 m/s.  All energy weights use the
same layout, demand, network initialization, and stochastic-channel seed within
each paired trial.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

from dso_optimization_extended import dso_optimize_extended


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results_round2" / "high_speed"
GAMMAS = (0.0, 1e-3)
SEED_BASE = 40928000


def run(*, smoke=False, trials=5, max_episodes=2000):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if smoke else "formal"
    record_path = OUTPUT / f"records_{tag}.jsonl"
    records = []
    for gamma in GAMMAS:
        for trial in range(trials):
            seed = SEED_BASE + trial
            started = time.perf_counter()
            result = dso_optimize_extended(
                num_users=4,
                area_size=10,
                h_unit=1,
                sigma=0.1,
                max_step=0.2,
                time_step_duration=0.1,
                max_episodes=max_episodes,
                use_allocation=True,
                shadow_std=0.15,
                rician_k=6.0,
                energy_weight=gamma,
                v_scale=100.0,
                seed=seed,
            )
            speeds = np.asarray(result["speed_sequence"], dtype=np.float64) * 100.0
            record = {
                "gamma": gamma,
                "trial": trial,
                "seed": seed,
                "time_steps": int(result["time_steps"]),
                "total_energy_j": float(result["total_energy"]),
                "training_time_s": float(result["total_training_time"]),
                "convergence_episode": int(result["convergence_episode"]),
                "mean_speed_mps": float(np.mean(speeds)),
                "p95_speed_mps": float(np.percentile(speeds, 95)),
                "max_speed_mps": float(np.max(speeds)),
                "wall_time_s": float(time.perf_counter() - started),
                "speed_sequence_mps": speeds,
                "trajectory": np.asarray(result["trajectory"], dtype=np.float64),
            }
            records.append(record)
            with record_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({k: v for k, v in record.items() if np.isscalar(v)}) + "\n")
            print(json.dumps({k: v for k, v in record.items() if np.isscalar(v)}), flush=True)

    np.savez_compressed(
        OUTPUT / f"high_speed_{tag}.npz",
        gamma=np.asarray([r["gamma"] for r in records]),
        trial=np.asarray([r["trial"] for r in records]),
        seed=np.asarray([r["seed"] for r in records]),
        time_steps=np.asarray([r["time_steps"] for r in records]),
        total_energy_j=np.asarray([r["total_energy_j"] for r in records]),
        training_time_s=np.asarray([r["training_time_s"] for r in records]),
        convergence_episode=np.asarray([r["convergence_episode"] for r in records]),
        mean_speed_mps=np.asarray([r["mean_speed_mps"] for r in records]),
        p95_speed_mps=np.asarray([r["p95_speed_mps"] for r in records]),
        max_speed_mps=np.asarray([r["max_speed_mps"] for r in records]),
        speed_sequence_mps=np.asarray([r["speed_sequence_mps"] for r in records], dtype=object),
        trajectory=np.asarray([r["trajectory"] for r in records], dtype=object),
        distance_scale_m=10.0,
        slot_duration_s=0.1,
        physical_speed_scale=100.0,
        maximum_speed_mps=20.0,
        shadow_std=0.15,
        rician_k=6.0,
        max_episodes=max_episodes,
    )
    summary_rows = []
    for gamma in GAMMAS:
        selected = [r for r in records if r["gamma"] == gamma]
        summary_rows.append({
            "gamma": gamma,
            "median_completion_slots": float(np.median([r["time_steps"] for r in selected])),
            "median_energy_j": float(np.median([r["total_energy_j"] for r in selected])),
            "median_mean_speed_mps": float(np.median([r["mean_speed_mps"] for r in selected])),
            "median_p95_speed_mps": float(np.median([r["p95_speed_mps"] for r in selected])),
            "completion_fraction": float(np.mean([r["time_steps"] < 500 for r in selected])),
        })
    summary = {
        "protocol": f"round2_high_speed_{tag}",
        "trials_per_gamma": trials,
        "paired_seeds": [SEED_BASE + i for i in range(trials)],
        "gammas": list(GAMMAS),
        "distance_scale_m_per_unit": 10.0,
        "slot_duration_s": 0.1,
        "maximum_speed_mps": 20.0,
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "rows": summary_rows,
    }
    (OUTPUT / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    run(
        smoke=args.smoke,
        trials=args.trials or (1 if args.smoke else 5),
        max_episodes=args.max_episodes or (8 if args.smoke else 2000),
    )

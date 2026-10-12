"""Paired L4V--SCA-QCQP solution-quality and direct-runtime experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dso_optimization_extended import dso_optimize_extended
from env_extended import compute_transmission_rate_stochastic, device, g


ROOT = Path(__file__).resolve().parent
QCQP = ROOT / "results_round2" / "qcqp" / "qcqp_formal.npz"
OUTPUT = ROOT / "results_round2" / "qcqp" / "l4v_paired_formal.npz"
SUMMARY = ROOT / "results_round2" / "qcqp" / "l4v_paired_summary.json"
RECORDS = ROOT / "results_round2" / "qcqp" / "l4v_paired_records.jsonl"
CHECKPOINTS = ROOT / "results_round2" / "qcqp" / "checkpoints"


def direct_runtime(model, users, tasks, warmups=5, repeats=30, horizon=500):
    """Time actual completed deterministic rollouts of a trained L4V policy."""
    users_t = torch.as_tensor(users, dtype=torch.float32, device=device)
    tasks_t = torch.as_tensor(tasks, dtype=torch.float32, device=device)
    h_unit = torch.tensor(1.0, device=device)
    sigma = torch.tensor(0.1, device=device)
    timings_ms = []
    steps = []
    model.eval()
    for repeat in range(warmups + repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            position = torch.zeros((1, 2), dtype=torch.float32, device=device)
            backlog = tasks_t.clone()
            completed_steps = horizon
            for step in range(horizon):
                angle, speed, allocation, _ = model(
                    position, users_t, backlog, use_allocation=False
                )
                position = g(position, angle, speed)
                rates = compute_transmission_rate_stochastic(
                    position,
                    users_t,
                    h_unit,
                    sigma,
                    shadow_std=0.0,
                    rician_k=None,
                )
                backlog = torch.clamp(
                    backlog - allocation * users_t.shape[0] * rates * 0.1,
                    min=0.0,
                )
                if bool(torch.all(backlog <= 1e-8)):
                    completed_steps = step + 1
                    break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        if repeat >= warmups:
            timings_ms.append(elapsed_ms)
            steps.append(completed_steps)
    values = np.asarray(timings_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "q25_ms": float(np.quantile(values, 0.25)),
        "q75_ms": float(np.quantile(values, 0.75)),
        "median_steps": float(np.median(steps)),
        "all_completed": bool(np.all(np.asarray(steps) < horizon)),
        "samples_ms": values,
    }


def run(trials=None, max_episodes=2000, smoke=False):
    with np.load(QCQP, allow_pickle=True) as raw:
        seeds = raw["seeds"].astype(np.int64)
        users = raw["user_positions"].astype(np.float32)
        tasks = raw["user_tasks"].astype(np.float32)
        qcqp_mission = raw["mission_time_slots"].astype(np.float64)
        qcqp_average = raw["average_completion_time_slots"].astype(np.float64)
        qcqp_solve = raw["solve_time_s"].astype(np.float64)
    if trials is not None:
        seeds, users, tasks = seeds[:trials], users[:trials], tasks[:trials]
        qcqp_mission = qcqp_mission[:trials]
        qcqp_average = qcqp_average[:trials]
        qcqp_solve = qcqp_solve[:trials]

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    records_path = RECORDS.with_name("l4v_paired_records_smoke.jsonl") if smoke else RECORDS
    records_path.write_text("", encoding="utf-8")
    records = []
    runtime_samples = []
    for index, seed in enumerate(seeds):
        started = time.perf_counter()
        result = dso_optimize_extended(
            num_users=4,
            area_size=10,
            h_unit=1,
            sigma=0.1,
            max_step=0.2,
            time_step_duration=0.1,
            max_episodes=max_episodes,
            use_allocation=False,
            shadow_std=0.0,
            rician_k=None,
            seed=int(seed),
            user_positions_override=users[index],
            initial_tasks_override=tasks[index],
            exact_evaluation=True,
            return_model=True,
        )
        model = result.pop("model")
        timing = direct_runtime(model, users[index], tasks[index])
        runtime_samples.append(timing["samples_ms"])
        checkpoint = CHECKPOINTS / f"l4v_qcqp_trial_{index + 1:02d}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": int(seed),
                "user_positions": users[index],
                "user_tasks": tasks[index],
                "use_allocation": False,
            },
            checkpoint,
        )
        record = {
            "trial": index,
            "seed": int(seed),
            "l4v_mission_time_slots": int(result["time_steps"]),
            "l4v_average_completion_time_slots": float(result["average_completion_time"]),
            "l4v_per_user_completion_times": result["per_user_completion_times"].tolist(),
            "l4v_training_time_s": float(result["total_training_time"]),
            "l4v_convergence_episode": int(result["convergence_episode"]),
            "l4v_direct_runtime_median_ms": timing["median_ms"],
            "l4v_direct_runtime_q25_ms": timing["q25_ms"],
            "l4v_direct_runtime_q75_ms": timing["q75_ms"],
            "l4v_direct_runtime_all_completed": timing["all_completed"],
            "qcqp_mission_time_slots": float(qcqp_mission[index]),
            "qcqp_average_completion_time_slots": float(qcqp_average[index]),
            "qcqp_solve_time_s": float(qcqp_solve[index]),
            "wall_time_s": float(time.perf_counter() - started),
            "trajectory": np.asarray(result["trajectory"], dtype=np.float64),
        }
        records.append(record)
        serializable = {
            key: value for key, value in record.items() if key != "trajectory"
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable) + "\n")
        print(json.dumps(serializable), flush=True)

    l4v_mission = np.asarray(
        [r["l4v_mission_time_slots"] for r in records], dtype=np.float64
    )
    l4v_average = np.asarray(
        [r["l4v_average_completion_time_slots"] for r in records], dtype=np.float64
    )
    l4v_training = np.asarray(
        [r["l4v_training_time_s"] for r in records], dtype=np.float64
    )
    l4v_runtime = np.asarray(
        [r["l4v_direct_runtime_median_ms"] for r in records], dtype=np.float64
    )
    output = OUTPUT.with_name("l4v_paired_smoke.npz") if smoke else OUTPUT
    summary_path = SUMMARY.with_name("l4v_paired_summary_smoke.json") if smoke else SUMMARY
    np.savez_compressed(
        output,
        seeds=seeds,
        user_positions=users,
        user_tasks=tasks,
        l4v_mission_time_slots=l4v_mission,
        l4v_average_completion_time_slots=l4v_average,
        l4v_per_user_completion_times=np.asarray(
            [r["l4v_per_user_completion_times"] for r in records], dtype=np.float64
        ),
        l4v_training_time_s=l4v_training,
        l4v_convergence_episode=np.asarray(
            [r["l4v_convergence_episode"] for r in records]
        ),
        l4v_direct_runtime_median_ms=l4v_runtime,
        l4v_direct_runtime_samples_ms=np.asarray(runtime_samples),
        l4v_trajectory=np.asarray([r["trajectory"] for r in records], dtype=object),
        qcqp_mission_time_slots=qcqp_mission,
        qcqp_average_completion_time_slots=qcqp_average,
        qcqp_solve_time_s=qcqp_solve,
        max_episodes=max_episodes,
    )
    summary = {
        "protocol": "round2_l4v_qcqp_paired_smoke" if smoke else "round2_l4v_qcqp_paired_formal",
        "trials": len(records),
        "seeds": seeds.tolist(),
        "median_l4v_mission_time_slots": float(np.median(l4v_mission)),
        "median_qcqp_mission_time_slots": float(np.median(qcqp_mission)),
        "median_l4v_average_completion_time_slots": float(np.median(l4v_average)),
        "median_qcqp_average_completion_time_slots": float(np.median(qcqp_average)),
        "median_l4v_training_time_s": float(np.median(l4v_training)),
        "median_qcqp_solve_time_s": float(np.median(qcqp_solve)),
        "median_l4v_direct_runtime_ms": float(np.median(l4v_runtime)),
        "l4v_faster_average_completion_fraction": float(np.mean(l4v_average < qcqp_average)),
        "l4v_not_slower_average_completion_fraction": float(np.mean(l4v_average <= qcqp_average)),
        "l4v_faster_mission_fraction": float(np.mean(l4v_mission < qcqp_mission)),
        "l4v_not_slower_mission_fraction": float(np.mean(l4v_mission <= qcqp_mission)),
        "all_direct_runtime_rollouts_completed": bool(
            all(r["l4v_direct_runtime_all_completed"] for r in records)
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.trials, args.max_episodes, args.smoke)

"""Reproduce the second-round runtime table and persist the timing protocol."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import torch

import run_latency as latency


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results_round2" / "runtime"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    users, tasks = latency.fixed_instance()
    h_unit = torch.tensor(latency.H_UNIT, device=latency.device)
    sigma = torch.tensor(latency.SIGMA, device=latency.device)

    from dqn_optimization import DQNAgent
    from a2c_optimization import A2CAgent
    from ddpg_optimization import DDPGAgent

    state_dim = 2 + latency.NUM_USERS * 2 + latency.NUM_USERS
    measured = {
        "L4V": latency.time_l4v_inference(users, tasks),
        "DQN": latency.time_agent_inference(
            lambda: DQNAgent(state_dim, 256, latency.NUM_USERS, latency.AREA,
                             h_unit, sigma, latency.V_MAX, latency.DT),
            False, users, tasks,
        ),
        "A2C": latency.time_agent_inference(
            lambda: A2CAgent(state_dim, 8, latency.NUM_USERS, latency.AREA,
                             h_unit, sigma, latency.V_MAX, latency.DT),
            True, users, tasks,
        ),
        "DDPG": latency.time_ddpg_inference(
            lambda: DDPGAgent(state_dim, 2, latency.NUM_USERS, latency.AREA,
                              h_unit, sigma, latency.V_MAX, latency.DT),
            users, tasks,
        ),
    }

    def median_csv(tag, column):
        frame = pd.read_csv(ROOT / "results" / "data" / f"{tag}_result_num_users_4.csv")
        return float(frame[column].median())

    steps = {method: median_csv(tag, "time_steps") for method, tag in
             [("L4V", "dso"), ("DQN", "dqn"), ("A2C", "a2c"), ("DDPG", "ddpg")]}
    execution_ms = {method: 1000.0 * measured[method] * steps[method] / latency.HORIZON
                    for method in measured}
    paired_l4v = json.loads(
        (ROOT / "results_round2" / "qcqp" / "l4v_paired_summary.json").read_text(
            encoding="utf-8"
        )
    )
    execution_ms["L4V"] = paired_l4v["median_l4v_direct_runtime_ms"]
    training_s = {method: median_csv(tag, "total_training_time") for method, tag in
                  [("L4V", "dso"), ("DQN", "dqn"), ("A2C", "a2c"), ("DDPG", "ddpg")]}
    solve_s = {
        "GA": median_csv("ga", "total_training_time"),
        "MPC": median_csv("mpc", "total_training_time"),
        "SCA-QCQP": json.loads(
            (ROOT / "results_round2" / "qcqp" / "summary_formal.json").read_text(encoding="utf-8")
        )["median_solve_time_s"],
    }
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    result = {
        "protocol": {
            "clock": "time.perf_counter wall clock",
            "warmup_rollouts": 5,
            "measured_rollouts": latency.N_REPEAT,
            "aggregation": (
                "L4V is the median of ten per-instance medians from 30 directly "
                "completed trained-policy rollouts; DQN/A2C/DDPG are warmed full-loop "
                "costs scaled to their observed median completion times"
            ),
            "cuda_synchronization": True,
            "processes": 1,
            "l4v_paired_instances": paired_l4v["trials"],
            "l4v_all_direct_rollouts_completed": paired_l4v[
                "all_direct_runtime_rollouts_completed"
            ],
        },
        "hardware": {
            "cpu": "12th Gen Intel Core i7-12700F, 12 cores, 20 logical processors",
            "memory_gb": 16,
            "gpu": gpu,
            "os": "Microsoft Windows 11 Professional, 64 bit",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "training_time_s": training_s,
        "execution_time_ms": execution_ms,
        "per_instance_solve_time_s": solve_s,
    }
    (OUTPUT / "runtime_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

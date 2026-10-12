"""Exact finite-action audit of the P3 surrogate against P1 on small tasks."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results_round2" / "surrogate_gap"
ACTIONS = np.asarray([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)


def enumerate_actions(horizon):
    codes = np.asarray(list(itertools.product(range(len(ACTIONS)), repeat=horizon)), dtype=np.int8)
    return ACTIONS[codes]


def evaluate(seed, action_sequences, *, step_size=0.5, dt=0.2, h_unit=1.0, sigma=0.1):
    rng = np.random.default_rng(seed)
    users = rng.uniform(-1.5, 1.5, size=(2, 2))
    tasks = rng.uniform(1.5, 2.5, size=2)
    paths = np.cumsum(action_sequences * step_size, axis=1)
    distance2 = np.sum((paths[:, :, None, :] - users[None, None, :, :]) ** 2, axis=-1)
    rates = np.log1p(h_unit / (sigma * np.maximum(distance2, 0.1)))
    backlog = np.maximum(tasks[None, None, :] - np.cumsum(rates * dt, axis=1), 0.0)
    complete = backlog <= 1e-9
    first = np.argmax(complete, axis=1) + 1
    never = ~np.any(complete, axis=1)
    first[never] = action_sequences.shape[1] + 1
    feasible = ~np.any(never, axis=1)
    p1 = np.mean(first, axis=1)
    p3 = np.sum(backlog, axis=(1, 2))
    feasible_indices = np.flatnonzero(feasible)
    if not len(feasible_indices):
        raise RuntimeError(f"No feasible sequence for seed {seed}")
    best_p1 = feasible_indices[np.argmin(p1[feasible_indices])]
    best_p3 = feasible_indices[np.argmin(p3[feasible_indices])]
    p1_star = float(p1[best_p1])
    p1_from_p3 = float(p1[best_p3])
    return {
        "seed": int(seed),
        "user_positions": users,
        "tasks": tasks,
        "p1_optimum": p1_star,
        "p1_at_p3_optimum": p1_from_p3,
        "absolute_gap": p1_from_p3 - p1_star,
        "relative_gap": (p1_from_p3 - p1_star) / p1_star,
        "p3_optimum": float(p3[best_p3]),
        "p3_at_p1_optimum": float(p3[best_p1]),
        "p1_action_sequence": action_sequences[best_p1],
        "p3_action_sequence": action_sequences[best_p3],
        "feasible_sequences": int(len(feasible_indices)),
    }


def run(trials=20, horizon=8, smoke=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    actions = enumerate_actions(horizon)
    seeds = np.arange(40927000, 40927000 + trials, dtype=np.int64)
    records = []
    for seed in seeds:
        record = evaluate(int(seed), actions)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if np.isscalar(v)}), flush=True)
    relative = np.asarray([r["relative_gap"] for r in records])
    absolute = np.asarray([r["absolute_gap"] for r in records])
    name = "smoke" if smoke else "formal"
    np.savez_compressed(
        OUTPUT / f"surrogate_gap_{name}.npz",
        seeds=seeds,
        user_positions=np.asarray([r["user_positions"] for r in records]),
        tasks=np.asarray([r["tasks"] for r in records]),
        p1_optimum=np.asarray([r["p1_optimum"] for r in records]),
        p1_at_p3_optimum=np.asarray([r["p1_at_p3_optimum"] for r in records]),
        absolute_gap=absolute,
        relative_gap=relative,
        p3_optimum=np.asarray([r["p3_optimum"] for r in records]),
        p3_at_p1_optimum=np.asarray([r["p3_at_p1_optimum"] for r in records]),
        p1_action_sequence=np.asarray([r["p1_action_sequence"] for r in records]),
        p3_action_sequence=np.asarray([r["p3_action_sequence"] for r in records]),
        feasible_sequences=np.asarray([r["feasible_sequences"] for r in records]),
        horizon=horizon,
        action_set=ACTIONS,
    )
    summary = {
        "protocol": f"round2_exact_small_action_{name}",
        "trials": trials,
        "horizon": horizon,
        "joint_action_sequences": int(len(actions)),
        "median_absolute_gap_slots": float(np.median(absolute)),
        "maximum_absolute_gap_slots": float(np.max(absolute)),
        "mean_relative_gap_percent": float(100 * np.mean(relative)),
        "maximum_relative_gap_percent": float(100 * np.max(relative)),
        "exact_match_fraction": float(np.mean(absolute == 0)),
    }
    (OUTPUT / f"summary_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    run(args.trials or (2 if args.smoke else 20), args.horizon, args.smoke)

"""Pretrain L4V across layouts and evaluate frozen-policy zero-shot transfer.

The broad comparison scripts optimize one policy per sampled layout.  This
experiment answers a different deployment question: can one policy be trained
on a distribution of layouts and then executed on held-out layouts without any
per-layout gradient update?

Every aggregate is backed by raw NPZ files containing held-out layouts,
trajectories, outcomes, optimization curves, timing, seeds, and metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from run_stress_tests import (
    DEVICE,
    NOMINAL_CHANNEL,
    FleetPolicy,
    advance,
    channel_rates,
    make_scenario,
    rollout,
)


@dataclass
class Config:
    updates: int
    batch_size: int
    train_horizon: int
    n_test: int
    n_eval: int
    eval_horizon: int
    tag: str


def config_from_args(args: argparse.Namespace) -> Config:
    if args.smoke:
        return Config(3, 1, 40, 2, 1, 40, "_smoke")
    if args.validate:
        return Config(160, 2, 160, 8, 2, 300, "_validate")
    return Config(800, 2, 220, 20, 3, 500, "")


def layout_loss(
    policy: FleetPolicy,
    seed: int,
    horizon: int,
    smoothness_weight: float = 0.01,
) -> torch.Tensor:
    scenario = make_scenario(seed)
    positions = scenario.start_positions.clone()
    backlog = scenario.tasks.clone()
    initial_total = torch.sum(backlog)
    objective = torch.zeros((), device=DEVICE)
    angles_history: list[torch.Tensor] = []

    for _ in range(horizon):
        angles, speeds, allocation = policy(
            positions,
            scenario.user_positions,
            backlog,
            scenario.obstacles,
        )
        positions = advance(positions, angles, speeds)
        rates = channel_rates(
            positions,
            scenario.user_positions,
            **NOMINAL_CHANNEL,
        )
        delivered = (
            allocation * scenario.tasks.numel() * rates * 0.1
        ).sum(dim=0)
        backlog = torch.clamp(backlog - delivered, min=0.0)
        objective = objective + torch.sum(backlog) / initial_total
        angles_history.append(angles)
        if float(torch.sum(backlog).detach()) < 1e-3 * backlog.numel():
            break

    if len(angles_history) > 1:
        angle_sequence = torch.stack(angles_history, dim=0)
        objective = objective + smoothness_weight * torch.mean(
            torch.diff(angle_sequence, dim=0) ** 2
        )
    return objective


def pretrain(config: Config, seed: int = 120_000) -> tuple[FleetPolicy, np.ndarray, float]:
    torch.manual_seed(seed)
    policy = FleetPolicy(num_users=4, num_aavs=1, num_obstacles=0).to(DEVICE)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    losses = np.zeros(config.updates)
    start = time.perf_counter()

    for update in range(config.updates):
        optimizer.zero_grad()
        loss = torch.zeros((), device=DEVICE)
        for batch_index in range(config.batch_size):
            layout_seed = seed + 1 + update * config.batch_size + batch_index
            loss = loss + layout_loss(
                policy, layout_seed, config.train_horizon
            )
        loss = loss / config.batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses[update] = float(loss.detach())
        if not np.isfinite(losses[update]):
            raise FloatingPointError("Non-finite pretraining objective.")
        if (update + 1) % max(1, config.updates // 10) == 0:
            print(
                f"pretrain {update + 1}/{config.updates}: "
                f"loss={losses[update]:.3f}",
                flush=True,
            )
    return policy, losses, time.perf_counter() - start


def evaluate(
    policy: FleetPolicy, config: Config, seed: int = 220_000
) -> dict[str, np.ndarray]:
    steps = np.zeros((config.n_test, config.n_eval))
    success = np.zeros_like(steps, dtype=bool)
    backlog = np.zeros_like(steps)
    inference = np.zeros_like(steps)
    trajectories = np.empty((config.n_test, config.n_eval), dtype=object)
    user_positions = np.zeros((config.n_test, 4, 2))
    tasks = np.zeros((config.n_test, 4))

    for test_index in range(config.n_test):
        scenario_seed = seed + test_index
        scenario = make_scenario(scenario_seed)
        user_positions[test_index] = (
            scenario.user_positions.detach().cpu().numpy()
        )
        tasks[test_index] = scenario.tasks.detach().cpu().numpy()
        for evaluation in range(config.n_eval):
            start = time.perf_counter()
            result = rollout(
                policy,
                scenario,
                config.eval_horizon,
                scenario_seed + 10_000 + evaluation,
            )
            inference[test_index, evaluation] = time.perf_counter() - start
            steps[test_index, evaluation] = result["steps"]
            success[test_index, evaluation] = result["completed"]
            backlog[test_index, evaluation] = result["final_backlog"]
            trajectories[test_index, evaluation] = result["trajectory"]
        print(
            f"held-out {test_index + 1}/{config.n_test}: "
            f"median={np.median(steps[test_index]):.0f}, "
            f"success={np.mean(success[test_index]):.2f}",
            flush=True,
        )
    return {
        "zero_shot_steps": steps,
        "zero_shot_success": success,
        "zero_shot_backlog": backlog,
        "zero_shot_inference_times": inference,
        "zero_shot_trajectories": trajectories,
        "zero_shot_user_positions": user_positions,
        "zero_shot_tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.validate:
        parser.error("--smoke and --validate are mutually exclusive")
    config = config_from_args(args)

    policy, losses, pretrain_time = pretrain(config)
    results = evaluate(policy, config)
    results.update(
        {
            "pretrain_losses": losses,
            "pretrain_wall_time": np.asarray(pretrain_time),
            "pretrain_layout_count": np.asarray(
                config.updates * config.batch_size
            ),
            "pretrain_updates": np.asarray(config.updates),
            "pretrain_batch_size": np.asarray(config.batch_size),
            "train_horizon": np.asarray(config.train_horizon),
            "eval_horizon": np.asarray(config.eval_horizon),
            "n_test": np.asarray(config.n_test),
            "n_eval": np.asarray(config.n_eval),
            "train_seed": np.asarray(120_000),
            "test_seed_start": np.asarray(220_000),
            "metadata": np.asarray(
                json.dumps(
                    {
                        "training_layouts_are_unique": True,
                        "test_layouts_disjoint_from_training": True,
                        "policy_frozen_during_test": True,
                        "device": str(DEVICE),
                        "torch_version": torch.__version__,
                        "channel": NOMINAL_CHANNEL,
                        "task_interval": [2.0, 4.0],
                        "area_size": 10.0,
                    }
                )
            ),
        }
    )

    output_dir = (
        Path(__file__).resolve().parent
        / "results_extended"
        / "generalization"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = output_dir / f"zero_shot_generalization{config.tag}.npz"
    np.savez_compressed(aggregate, **results)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for test_index in range(config.n_test):
        np.savez_compressed(
            raw_dir
            / f"zero_shot{config.tag}_trial{test_index + 1:02d}.npz",
            solutions=results["zero_shot_trajectories"][test_index],
            objective_values=np.column_stack(
                [
                    results["zero_shot_steps"][test_index],
                    results["zero_shot_backlog"][test_index],
                ]
            ),
            wall_times=results["zero_shot_inference_times"][test_index],
            user_positions=results["zero_shot_user_positions"][test_index],
            tasks=results["zero_shot_tasks"][test_index],
            convergence_curves=losses,
            seed=220_000 + test_index,
            metadata=results["metadata"],
        )
    print(f"Saved {aggregate}")
    if args.smoke:
        required = {
            "solutions",
            "objective_values",
            "wall_times",
            "convergence_curves",
            "metadata",
        }
        sample = np.load(
            raw_dir / "zero_shot_smoke_trial01.npz", allow_pickle=True
        )
        missing = required.difference(sample.files)
        if missing:
            raise RuntimeError(f"Smoke artifact missing keys: {sorted(missing)}")
        print("Smoke test PASSED")


if __name__ == "__main__":
    main()

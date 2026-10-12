"""Reviewer-driven stress tests for L4V.

This script extends the executed differentiable simulator in five directions:

1. channel-generator mismatch at deployment;
2. noisy user locations and partial observations of remaining backlogs;
3. two-AAV cooperation under a fleet-wide fixed bandwidth budget; and
4. circular no-fly regions enforced by a differentiable clearance barrier; and
5. sensitivity to the heading-smoothness coefficient.

The experiments are simulation studies.  Every reported point is produced by
an executed policy rollout, and all raw trajectories, losses, timings, seeds,
and metadata are saved in compressed NPZ files.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOMINAL_CHANNEL = {"shadow_std": 0.15, "rician_k": 6.0, "pathloss_exp": 2.0}
OBSTACLE_CLEARANCE_MARGIN = 0.20
OBSTACLE_BARRIER_WEIGHT = 250.0
CHANNEL_CONDITIONS = [
    ("Matched", 0.15, 6.0, 2.0),
    ("Mild", 0.25, 3.0, 2.2),
    ("Severe", 0.40, 0.0, 2.5),
]
OBSERVATION_CONDITIONS = [
    ("Full", 0.0, 0.0),
    ("Location noise", 0.30, 0.0),
    ("50% backlog mask", 0.0, 0.50),
    ("Both", 0.30, 0.50),
]


@dataclass
class Config:
    n_trials: int
    n_eval: int
    train_episodes: int
    train_horizon: int
    eval_horizon: int
    tag: str


@dataclass
class Scenario:
    user_positions: torch.Tensor
    tasks: torch.Tensor
    start_positions: torch.Tensor
    obstacles: torch.Tensor
    area_size: float = 10.0


class FleetPolicy(nn.Module):
    """Shared policy for one or more AAVs with fleet-wide radio allocation."""

    def __init__(self, num_users: int, num_aavs: int, num_obstacles: int):
        super().__init__()
        self.num_users = num_users
        self.num_aavs = num_aavs
        self.num_obstacles = num_obstacles
        input_dim = (
            2 * num_aavs
            + 2 * num_aavs * num_users
            + num_users
            + 3 * num_obstacles
        )
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.angle_head = nn.Linear(32, num_aavs)
        self.speed_head = nn.Linear(32, num_aavs)
        self.alloc_head = nn.Linear(32, num_aavs * num_users)
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(layer.weight)
        for layer in (self.angle_head, self.speed_head, self.alloc_head):
            nn.init.uniform_(layer.weight, -0.01, 0.01)

    def forward(
        self,
        aav_positions: torch.Tensor,
        observed_users: torch.Tensor,
        observed_backlog: torch.Tensor,
        obstacles: torch.Tensor,
        max_step: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relative_users = (
            observed_users.unsqueeze(0) - aav_positions.unsqueeze(1)
        )
        components = [
            aav_positions.flatten(),
            relative_users.flatten(),
            observed_backlog.flatten(),
        ]
        if self.num_obstacles:
            components.append(obstacles.flatten())
        x = torch.cat(components)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        angles = torch.tanh(self.angle_head(x)) * math.pi
        speeds = torch.sigmoid(self.speed_head(x)) * max_step
        allocation = F.softmax(self.alloc_head(x), dim=-1).view(
            self.num_aavs, self.num_users
        )
        return angles, speeds, allocation


def make_scenario(
    seed: int,
    num_users: int = 4,
    num_aavs: int = 1,
    obstacle_case: bool = False,
) -> Scenario:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    if obstacle_case:
        x = 1.5 + 3.0 * torch.rand(num_users, generator=generator, device=DEVICE)
        y = -3.5 + 7.0 * torch.rand(num_users, generator=generator, device=DEVICE)
        user_positions = torch.stack([x, y], dim=1)
        start_positions = torch.tensor([[-4.0, 0.0]], device=DEVICE)
        obstacles = torch.tensor([[0.0, 0.0, 1.35]], device=DEVICE)
    else:
        user_positions = (
            torch.rand((num_users, 2), generator=generator, device=DEVICE) - 0.5
        ) * 10.0
        if num_aavs == 1:
            start_positions = torch.zeros((1, 2), device=DEVICE)
        else:
            offsets = torch.linspace(
                -0.35, 0.35, num_aavs, device=DEVICE
            )
            start_positions = torch.stack(
                [offsets, torch.zeros_like(offsets)], dim=1
            )
        obstacles = torch.empty((0, 3), device=DEVICE)
    tasks = (
        2.0
        + 2.0 * torch.rand(num_users, generator=generator, device=DEVICE)
    )
    return Scenario(user_positions, tasks, start_positions, obstacles)


def channel_rates(
    aav_positions: torch.Tensor,
    user_positions: torch.Tensor,
    shadow_std: float,
    rician_k: float,
    pathloss_exp: float,
    sigma: float = 0.1,
    h_unit: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return an M-by-N normalized spectral-efficiency matrix."""
    horizontal_sq = torch.sum(
        (aav_positions.unsqueeze(1) - user_positions.unsqueeze(0)) ** 2,
        dim=-1,
    )
    horizontal_sq = torch.clamp(horizontal_sq, min=0.1)
    path_gain = h_unit / horizontal_sq.pow(pathloss_exp / 2.0)
    shape = horizontal_sq.shape
    shadow = torch.exp(
        shadow_std
        * torch.randn(
            shape, generator=generator, device=aav_positions.device
        )
    )
    los_amp = math.sqrt(rician_k / (rician_k + 1.0))
    scatter_std = math.sqrt(1.0 / (2.0 * (rician_k + 1.0)))
    eps_r = torch.randn(
        shape, generator=generator, device=aav_positions.device
    )
    eps_i = torch.randn(
        shape, generator=generator, device=aav_positions.device
    )
    fading = (los_amp + scatter_std * eps_r) ** 2 + (
        scatter_std * eps_i
    ) ** 2
    return torch.log1p(path_gain * shadow * fading / sigma)


def advance(
    positions: torch.Tensor, angles: torch.Tensor, speeds: torch.Tensor
) -> torch.Tensor:
    movement = torch.stack(
        [torch.cos(angles) * speeds, torch.sin(angles) * speeds], dim=1
    )
    return positions + movement


def constraint_penalty(
    positions: torch.Tensor,
    obstacles: torch.Tensor,
    obstacle_weight: float,
    min_separation: float = 0.45,
) -> torch.Tensor:
    penalty = torch.zeros((), device=positions.device)
    if obstacles.numel() and obstacle_weight > 0:
        centres = obstacles[:, :2]
        radii = obstacles[:, 2]
        distances = torch.linalg.vector_norm(
            positions.unsqueeze(1) - centres.unsqueeze(0), dim=-1
        )
        clearance = distances - radii.unsqueeze(0)
        # Penalize entry into a safety buffer so that optimization does not
        # merely graze the hard no-fly boundary used at evaluation.
        soft_violation = (
            F.softplus(12.0 * (OBSTACLE_CLEARANCE_MARGIN - clearance)) / 12.0
        )
        penalty = penalty + obstacle_weight * torch.sum(soft_violation**2)
    if positions.shape[0] > 1:
        pairwise = torch.cdist(positions, positions)
        mask = torch.triu(
            torch.ones_like(pairwise, dtype=torch.bool), diagonal=1
        )
        separation = pairwise[mask]
        soft_collision = F.softplus(
            12.0 * (min_separation - separation)
        ) / 12.0
        penalty = penalty + 30.0 * torch.sum(soft_collision**2)
    boundary = torch.clamp(torch.abs(positions) - 5.0, min=0.0)
    penalty = penalty + 0.1 * torch.sum(boundary)
    return penalty


def train_policy(
    scenario: Scenario,
    num_aavs: int,
    episodes: int,
    horizon: int,
    seed: int,
    obstacle_weight: float = 0.0,
    smoothness_weight: float = 0.01,
    observe_obstacles: bool = True,
) -> tuple[FleetPolicy, np.ndarray, float]:
    torch.manual_seed(seed)
    policy = FleetPolicy(
        scenario.tasks.numel(),
        num_aavs,
        scenario.obstacles.shape[0] if observe_obstacles else 0,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    losses: list[float] = []
    start_time = time.perf_counter()
    initial_total = torch.sum(scenario.tasks)
    policy_obstacles = (
        scenario.obstacles
        if observe_obstacles
        else torch.empty((0, 3), device=DEVICE)
    )

    for _ in range(episodes):
        positions = scenario.start_positions.clone()
        backlog = scenario.tasks.clone()
        objective = torch.zeros((), device=DEVICE)
        angle_history: list[torch.Tensor] = []
        for _step in range(horizon):
            angles, speeds, allocation = policy(
                positions,
                scenario.user_positions,
                backlog,
                policy_obstacles,
            )
            positions = advance(positions, angles, speeds)
            rates = channel_rates(
                positions,
                scenario.user_positions,
                **NOMINAL_CHANNEL,
            )
            # N is the fixed fleet-wide resource budget.  For M=1 and
            # a_i=1/N, this reduces exactly to the original per-user rate.
            delivered = (
                allocation
                * scenario.tasks.numel()
                * rates
                * 0.1
            ).sum(dim=0)
            backlog = torch.clamp(backlog - delivered, min=0.0)
            objective = objective + torch.sum(backlog) / initial_total
            objective = objective + constraint_penalty(
                positions, scenario.obstacles, obstacle_weight
            )
            angle_history.append(angles)
            if float(torch.sum(backlog).detach()) < 1e-3 * backlog.numel():
                break

        if len(angle_history) > 1 and smoothness_weight > 0:
            angle_sequence = torch.stack(angle_history, dim=0)
            objective = objective + smoothness_weight * torch.mean(
                torch.diff(angle_sequence, dim=0) ** 2
            )

        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        loss_value = float(objective.detach())
        if not np.isfinite(loss_value):
            raise FloatingPointError("Non-finite training objective.")
        losses.append(loss_value)

    return policy, np.asarray(losses), time.perf_counter() - start_time


def rollout(
    policy: FleetPolicy,
    scenario: Scenario,
    horizon: int,
    seed: int,
    channel: dict[str, float] | None = None,
    location_noise: float = 0.0,
    backlog_mask_probability: float = 0.0,
) -> dict[str, np.ndarray | float | int | bool]:
    channel_generator = torch.Generator(device=DEVICE)
    channel_generator.manual_seed(seed)
    observation_generator = torch.Generator(device=DEVICE)
    observation_generator.manual_seed(seed + 1_000_000)
    channel = NOMINAL_CHANNEL if channel is None else channel
    positions = scenario.start_positions.clone()
    backlog = scenario.tasks.clone()
    observed_backlog = backlog.clone()
    user_bias = location_noise * torch.randn(
        scenario.user_positions.shape,
        generator=observation_generator,
        device=DEVICE,
    )
    policy_obstacles = (
        scenario.obstacles
        if policy.num_obstacles
        else torch.empty((0, 3), device=DEVICE)
    )
    trajectory = [positions.detach().cpu().numpy().copy()]
    allocations = []
    collision = False
    minimum_clearance = float("inf")
    completion_step = horizon

    with torch.no_grad():
        for step in range(horizon):
            observed_users = scenario.user_positions + user_bias
            visible_backlogs = (
                torch.rand(
                    backlog.shape,
                    generator=observation_generator,
                    device=DEVICE,
                )
                >= backlog_mask_probability
            )
            observed_backlog = torch.where(
                visible_backlogs, backlog, observed_backlog
            )
            angles, speeds, allocation = policy(
                positions, observed_users, observed_backlog, policy_obstacles
            )
            positions = advance(positions, angles, speeds)
            rates = channel_rates(
                positions,
                scenario.user_positions,
                **channel,
                generator=channel_generator,
            )
            delivered = (
                allocation * scenario.tasks.numel() * rates * 0.1
            ).sum(dim=0)
            backlog = torch.clamp(backlog - delivered, min=0.0)
            trajectory.append(positions.detach().cpu().numpy().copy())
            allocations.append(allocation.detach().cpu().numpy().copy())

            if scenario.obstacles.numel():
                distances = torch.linalg.vector_norm(
                    positions.unsqueeze(1)
                    - scenario.obstacles[:, :2].unsqueeze(0),
                    dim=-1,
                )
                clearance = distances - scenario.obstacles[:, 2].unsqueeze(0)
                minimum_clearance = min(
                    minimum_clearance, float(torch.min(clearance))
                )
                collision = collision or bool(torch.any(clearance < 0))
            if float(torch.sum(backlog)) < 1e-3 * backlog.numel():
                completion_step = step + 1
                break

    trajectory_array = np.asarray(trajectory)
    step_distances = np.linalg.norm(
        np.diff(trajectory_array, axis=0), axis=-1
    )
    displacement = np.diff(trajectory_array, axis=0)
    headings = np.arctan2(displacement[..., 1], displacement[..., 0])
    if headings.shape[0] > 1:
        turn_variation = float(
            np.mean(1.0 - np.cos(np.diff(headings, axis=0)))
        )
    else:
        turn_variation = 0.0
    completed = float(torch.sum(backlog)) < 1e-3 * backlog.numel()
    safe_success = completed and not collision
    return {
        "steps": completion_step,
        "completed": completed,
        "safe_success": safe_success,
        "collision": collision,
        "final_backlog": float(torch.sum(backlog)),
        "minimum_clearance": minimum_clearance,
        "fleet_distance": float(np.sum(step_distances)),
        "turn_variation": turn_variation,
        "trajectory": trajectory_array,
        "allocations": np.asarray(allocations),
    }


def save_trial(
    raw_dir: Path,
    name: str,
    trial: int,
    seed: int,
    scenario: Scenario,
    losses: np.ndarray,
    train_wall_time: float,
    rollouts: list[dict],
    metadata: dict,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_dir / f"{name}_trial{trial:02d}.npz",
        solutions=np.asarray(
            [item["trajectory"] for item in rollouts], dtype=object
        ),
        objective_values=np.asarray(
            [[item["steps"], item["final_backlog"]] for item in rollouts],
            dtype=float,
        ),
        convergence_curves=losses,
        wall_times=np.asarray([train_wall_time], dtype=float),
        user_positions=scenario.user_positions.detach().cpu().numpy(),
        tasks=scenario.tasks.detach().cpu().numpy(),
        start_positions=scenario.start_positions.detach().cpu().numpy(),
        obstacles=scenario.obstacles.detach().cpu().numpy(),
        seed=seed,
        metadata=json.dumps(metadata),
    )


def run_robustness(config: Config, output_dir: Path) -> dict[str, np.ndarray]:
    channel_steps = np.zeros(
        (config.n_trials, len(CHANNEL_CONDITIONS), config.n_eval)
    )
    channel_success = np.zeros_like(channel_steps, dtype=bool)
    channel_backlog = np.zeros_like(channel_steps)
    observation_steps = np.zeros(
        (config.n_trials, len(OBSERVATION_CONDITIONS), config.n_eval)
    )
    observation_success = np.zeros_like(observation_steps, dtype=bool)
    observation_backlog = np.zeros_like(observation_steps)
    train_times = np.zeros(config.n_trials)
    loss_curves = np.empty(config.n_trials, dtype=object)
    all_trajectories = np.empty(
        (config.n_trials, len(CHANNEL_CONDITIONS) + len(OBSERVATION_CONDITIONS)),
        dtype=object,
    )

    for trial in range(config.n_trials):
        seed = 71000 + trial
        scenario = make_scenario(seed)
        policy, losses, train_time = train_policy(
            scenario,
            num_aavs=1,
            episodes=config.train_episodes,
            horizon=config.train_horizon,
            seed=seed,
        )
        train_times[trial] = train_time
        loss_curves[trial] = losses
        saved_rollouts: list[dict] = []
        for condition_index, (label, shadow, rician, exponent) in enumerate(
            CHANNEL_CONDITIONS
        ):
            for evaluation in range(config.n_eval):
                result = rollout(
                    policy,
                    scenario,
                    config.eval_horizon,
                    seed + 1000 + evaluation,
                    channel={
                        "shadow_std": shadow,
                        "rician_k": rician,
                        "pathloss_exp": exponent,
                    },
                )
                channel_steps[trial, condition_index, evaluation] = result["steps"]
                channel_success[trial, condition_index, evaluation] = result[
                    "completed"
                ]
                channel_backlog[trial, condition_index, evaluation] = result[
                    "final_backlog"
                ]
                saved_rollouts.append(result)
                if evaluation == 0:
                    all_trajectories[trial, condition_index] = result["trajectory"]

        offset = len(CHANNEL_CONDITIONS)
        for condition_index, (label, loc_noise, backlog_mask) in enumerate(
            OBSERVATION_CONDITIONS
        ):
            for evaluation in range(config.n_eval):
                result = rollout(
                    policy,
                    scenario,
                    config.eval_horizon,
                    seed + 10000 + evaluation,
                    location_noise=loc_noise,
                    backlog_mask_probability=backlog_mask,
                )
                observation_steps[trial, condition_index, evaluation] = result[
                    "steps"
                ]
                observation_success[trial, condition_index, evaluation] = result[
                    "completed"
                ]
                observation_backlog[trial, condition_index, evaluation] = result[
                    "final_backlog"
                ]
                saved_rollouts.append(result)
                if evaluation == 0:
                    all_trajectories[trial, offset + condition_index] = result[
                        "trajectory"
                    ]
        save_trial(
            output_dir / "raw",
            f"robustness{config.tag}",
            trial + 1,
            seed,
            scenario,
            losses,
            train_time,
            saved_rollouts,
            {
                "channel_conditions": CHANNEL_CONDITIONS,
                "observation_conditions": OBSERVATION_CONDITIONS,
                "train_horizon": config.train_horizon,
                "eval_horizon": config.eval_horizon,
                "train_episodes": config.train_episodes,
            },
        )
        print(
            f"robustness trial {trial + 1}/{config.n_trials}: "
            f"matched={np.median(channel_steps[trial, 0]):.0f}, "
            f"severe={np.median(channel_steps[trial, -1]):.0f}, "
            f"partial={np.median(observation_steps[trial, -1]):.0f}",
            flush=True,
        )

    return {
        "channel_steps": channel_steps,
        "channel_success": channel_success,
        "channel_backlog": channel_backlog,
        "observation_steps": observation_steps,
        "observation_success": observation_success,
        "observation_backlog": observation_backlog,
        "robust_train_times": train_times,
        "robust_loss_curves": loss_curves,
        "robust_trajectories": all_trajectories,
    }


def run_multi_aav(config: Config, output_dir: Path) -> dict[str, np.ndarray]:
    aav_counts = [1, 2]
    steps = np.zeros((config.n_trials, len(aav_counts), config.n_eval))
    success = np.zeros_like(steps, dtype=bool)
    fleet_distance = np.zeros_like(steps)
    train_times = np.zeros((config.n_trials, len(aav_counts)))
    loss_curves = np.empty((config.n_trials, len(aav_counts)), dtype=object)
    trajectories = np.empty((config.n_trials, len(aav_counts)), dtype=object)

    for trial in range(config.n_trials):
        seed = 81000 + trial
        for count_index, num_aavs in enumerate(aav_counts):
            scenario = make_scenario(seed, num_aavs=num_aavs)
            policy, losses, train_time = train_policy(
                scenario,
                num_aavs=num_aavs,
                episodes=config.train_episodes,
                horizon=config.train_horizon,
                seed=seed + 100 * num_aavs,
            )
            train_times[trial, count_index] = train_time
            loss_curves[trial, count_index] = losses
            saved_rollouts = []
            for evaluation in range(config.n_eval):
                result = rollout(
                    policy,
                    scenario,
                    config.eval_horizon,
                    seed + 1000 * num_aavs + evaluation,
                )
                steps[trial, count_index, evaluation] = result["steps"]
                success[trial, count_index, evaluation] = result["completed"]
                fleet_distance[trial, count_index, evaluation] = result[
                    "fleet_distance"
                ]
                saved_rollouts.append(result)
                if evaluation == 0:
                    trajectories[trial, count_index] = result["trajectory"]
            save_trial(
                output_dir / "raw",
                f"multi_aav_{num_aavs}{config.tag}",
                trial + 1,
                seed,
                scenario,
                losses,
                train_time,
                saved_rollouts,
                {
                    "num_aavs": num_aavs,
                    "fleet_wide_resource_budget": scenario.tasks.numel(),
                    "train_horizon": config.train_horizon,
                    "eval_horizon": config.eval_horizon,
                    "train_episodes": config.train_episodes,
                },
            )
        print(
            f"multi-AAV trial {trial + 1}/{config.n_trials}: "
            f"one={np.median(steps[trial, 0]):.0f}, "
            f"two={np.median(steps[trial, 1]):.0f}",
            flush=True,
        )
    return {
        "multi_steps": steps,
        "multi_success": success,
        "multi_fleet_distance": fleet_distance,
        "multi_train_times": train_times,
        "multi_loss_curves": loss_curves,
        "multi_trajectories": trajectories,
    }


def run_obstacles(config: Config, output_dir: Path) -> dict[str, np.ndarray]:
    labels = ["Unaware", "Aware"]
    weights = [0.0, OBSTACLE_BARRIER_WEIGHT]
    steps = np.zeros((config.n_trials, 2, config.n_eval))
    safe_success = np.zeros_like(steps, dtype=bool)
    collisions = np.zeros_like(steps, dtype=bool)
    clearances = np.zeros_like(steps)
    train_times = np.zeros((config.n_trials, 2))
    loss_curves = np.empty((config.n_trials, 2), dtype=object)
    trajectories = np.empty((config.n_trials, 2), dtype=object)
    scenarios = np.empty(config.n_trials, dtype=object)

    for trial in range(config.n_trials):
        seed = 91000 + trial
        scenario = make_scenario(seed, obstacle_case=True)
        scenarios[trial] = {
            "users": scenario.user_positions.detach().cpu().numpy(),
            "tasks": scenario.tasks.detach().cpu().numpy(),
            "start": scenario.start_positions.detach().cpu().numpy(),
            "obstacles": scenario.obstacles.detach().cpu().numpy(),
        }
        for method_index, weight in enumerate(weights):
            policy, losses, train_time = train_policy(
                scenario,
                num_aavs=1,
                episodes=config.train_episodes,
                horizon=config.train_horizon,
                seed=seed,
                obstacle_weight=weight,
                observe_obstacles=bool(method_index),
            )
            train_times[trial, method_index] = train_time
            loss_curves[trial, method_index] = losses
            saved_rollouts = []
            for evaluation in range(config.n_eval):
                result = rollout(
                    policy,
                    scenario,
                    config.eval_horizon,
                    seed + 1000 + evaluation,
                )
                safe_steps = (
                    result["steps"]
                    if result["safe_success"]
                    else config.eval_horizon
                )
                steps[trial, method_index, evaluation] = safe_steps
                safe_success[trial, method_index, evaluation] = result[
                    "safe_success"
                ]
                collisions[trial, method_index, evaluation] = result["collision"]
                clearances[trial, method_index, evaluation] = result[
                    "minimum_clearance"
                ]
                saved_rollouts.append(result)
                if evaluation == 0:
                    trajectories[trial, method_index] = result["trajectory"]
            save_trial(
                output_dir / "raw",
                f"obstacle_{labels[method_index].lower()}{config.tag}",
                trial + 1,
                seed,
                scenario,
                losses,
                train_time,
                saved_rollouts,
                {
                    "method": labels[method_index],
                    "obstacle_weight": weight,
                    "obstacle_clearance_margin": OBSTACLE_CLEARANCE_MARGIN,
                    "observes_obstacle_geometry": bool(method_index),
                    "train_horizon": config.train_horizon,
                    "eval_horizon": config.eval_horizon,
                    "train_episodes": config.train_episodes,
                },
            )
        print(
            f"obstacle trial {trial + 1}/{config.n_trials}: "
            f"unaware_safe={safe_success[trial, 0].mean():.2f}, "
            f"aware_safe={safe_success[trial, 1].mean():.2f}",
            flush=True,
        )
    return {
        "obstacle_steps": steps,
        "obstacle_safe_success": safe_success,
        "obstacle_collisions": collisions,
        "obstacle_clearances": clearances,
        "obstacle_train_times": train_times,
        "obstacle_loss_curves": loss_curves,
        "obstacle_trajectories": trajectories,
        "obstacle_scenarios": scenarios,
    }


def run_smoothness(config: Config, output_dir: Path) -> dict[str, np.ndarray]:
    weights = np.asarray([0.0, 0.0025, 0.01, 0.04])
    episodes = min(config.train_episodes, 120)
    steps = np.zeros((config.n_trials, len(weights), config.n_eval))
    success = np.zeros_like(steps, dtype=bool)
    turn_variation = np.zeros_like(steps)
    train_times = np.zeros((config.n_trials, len(weights)))
    loss_curves = np.empty((config.n_trials, len(weights)), dtype=object)
    trajectories = np.empty((config.n_trials, len(weights)), dtype=object)

    for trial in range(config.n_trials):
        seed = 101000 + trial
        scenario = make_scenario(seed)
        for weight_index, weight in enumerate(weights):
            policy, losses, train_time = train_policy(
                scenario,
                num_aavs=1,
                episodes=episodes,
                horizon=config.train_horizon,
                seed=seed,
                smoothness_weight=float(weight),
            )
            train_times[trial, weight_index] = train_time
            loss_curves[trial, weight_index] = losses
            saved_rollouts = []
            for evaluation in range(config.n_eval):
                result = rollout(
                    policy,
                    scenario,
                    config.eval_horizon,
                    seed + 1000 * weight_index + evaluation,
                )
                steps[trial, weight_index, evaluation] = result["steps"]
                success[trial, weight_index, evaluation] = result["completed"]
                turn_variation[trial, weight_index, evaluation] = result[
                    "turn_variation"
                ]
                saved_rollouts.append(result)
                if evaluation == 0:
                    trajectories[trial, weight_index] = result["trajectory"]
            save_trial(
                output_dir / "raw",
                f"smoothness_{weight:g}{config.tag}",
                trial + 1,
                seed,
                scenario,
                losses,
                train_time,
                saved_rollouts,
                {
                    "smoothness_weight": float(weight),
                    "train_horizon": config.train_horizon,
                    "eval_horizon": config.eval_horizon,
                    "train_episodes": episodes,
                },
            )
        print(
            f"smoothness trial {trial + 1}/{config.n_trials}: "
            f"steps={np.median(steps[trial], axis=1).astype(int).tolist()}",
            flush=True,
        )
    return {
        "smoothness_weights": weights,
        "smoothness_steps": steps,
        "smoothness_success": success,
        "smoothness_turn_variation": turn_variation,
        "smoothness_train_times": train_times,
        "smoothness_loss_curves": loss_curves,
        "smoothness_trajectories": trajectories,
    }


def get_config(args: argparse.Namespace) -> Config:
    if args.smoke:
        return Config(1, 1, 3, 40, 40, "_smoke")
    if args.validate:
        return Config(2, 2, 120, 220, 220, "_validate")
    return Config(5, 5, 180, 220, 500, "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        choices=["robustness", "multi", "obstacle", "smoothness", "all"],
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.validate:
        parser.error("--smoke and --validate are mutually exclusive")

    config = get_config(args)
    output_dir = Path(__file__).resolve().parent / "results_extended" / "stress"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, np.ndarray] = {}
    if args.part in ("robustness", "all"):
        results.update(run_robustness(config, output_dir))
    if args.part in ("multi", "all"):
        results.update(run_multi_aav(config, output_dir))
    if args.part in ("obstacle", "all"):
        results.update(run_obstacles(config, output_dir))
    if args.part in ("smoothness", "all"):
        results.update(run_smoothness(config, output_dir))

    results.update(
        {
            "channel_condition_labels": np.asarray(
                [item[0] for item in CHANNEL_CONDITIONS]
            ),
            "observation_condition_labels": np.asarray(
                [item[0] for item in OBSERVATION_CONDITIONS]
            ),
            "channel_conditions": np.asarray(
                [item[1:] for item in CHANNEL_CONDITIONS], dtype=float
            ),
            "observation_conditions": np.asarray(
                [item[1:] for item in OBSERVATION_CONDITIONS], dtype=float
            ),
            "n_trials": np.asarray(config.n_trials),
            "n_eval": np.asarray(config.n_eval),
            "train_episodes": np.asarray(config.train_episodes),
            "train_horizon": np.asarray(config.train_horizon),
            "eval_horizon": np.asarray(config.eval_horizon),
            "device": np.asarray(str(DEVICE)),
        }
    )
    part_token = "" if args.part == "all" else f"_{args.part}"
    output_path = output_dir / f"stress_suite{part_token}{config.tag}.npz"
    np.savez_compressed(output_path, **results)
    print(f"Saved {output_path}")
    if args.smoke:
        print("Smoke test PASSED")
    elif args.validate:
        print("Validation run PASSED")


if __name__ == "__main__":
    main()

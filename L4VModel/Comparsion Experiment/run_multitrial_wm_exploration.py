"""Validation of world-model-guided multi-trial action exploration.

The experiment separates four effects under a hidden channel-model error:

1. Single trial: one action trajectory through one nominal world model.
2. Ensemble only: one action trajectory evaluated by a world-model ensemble.
3. Random multi-trial: several perturbed action trajectories, averaged.
4. Counterfactual P/N: the same action branches are scored by the ensemble;
   robustly positive and negative branches provide contrastive supervision.

All methods use the same scenarios, initial policy parameters, optimizer,
number of parameter updates, and hidden true channel.  The true channel is
used only for evaluation and for retrospective label auditing; it is never
used by an optimizer.  Raw per-trial arrays, policy vectors, convergence
curves, timing, metadata, figures, and a Markdown table are saved.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_stress_tests import DEVICE, FleetPolicy, Scenario, advance


METHODS = (
    "Single trial",
    "Ensemble only",
    "Random multi-trial",
    "Counterfactual P/N",
)
METHOD_COLORS = {
    "Single trial": "#7f7f7f",
    "Ensemble only": "#1f77b4",
    "Random multi-trial": "#ff7f0e",
    "Counterfactual P/N": "#d62728",
}


@dataclass(frozen=True)
class WorldModelSpec:
    pathloss_exp: float
    shadow_strength: float
    shadow_width: float


# The direct route lies on the symmetry axis of an unobserved attenuation
# region. A single deterministic rollout has no transverse gradient there.
# The hidden field is stronger than the nominal WM, while the ensemble covers
# plausible strengths without seeing the hidden true specification.
NOMINAL_MODEL = WorldModelSpec(2.0, 0.45, 0.80)
ENSEMBLE_MODELS = (
    WorldModelSpec(2.0, 1.00, 1.00),
    WorldModelSpec(2.0, 3.00, 1.40),
    WorldModelSpec(2.0, 5.00, 1.80),
)
TRUE_MODEL = WorldModelSpec(2.0, 4.00, 1.50)
SIGMA = 0.10
MAX_STEP = 0.20
TIME_STEP = 0.10
ROBUSTNESS_WEIGHT = 0.15
POSITIVE_WEIGHT = 60.00
NEGATIVE_WEIGHT = 0.00
DISTANCE_MARGIN = 0.035


@dataclass(frozen=True)
class ExperimentConfig:
    n_trials: int
    train_episodes: int
    train_horizon: int
    eval_horizon: int
    n_branches: int
    checkpoint_every: int
    tag: str


@dataclass
class ActionNoise:
    angle: torch.Tensor
    speed: torch.Tensor
    allocation: torch.Tensor


def channel_rates_parametric(
    aav_positions: torch.Tensor,
    user_positions: torch.Tensor,
    model: WorldModelSpec,
) -> torch.Tensor:
    """Deterministic rate with an action-dependent hidden shadow field."""
    horizontal_sq = torch.sum(
        (aav_positions.unsqueeze(1) - user_positions.unsqueeze(0)) ** 2,
        dim=-1,
    )
    horizontal_sq = torch.clamp(horizontal_sq, min=0.1)
    path_gain = horizontal_sq.pow(-model.pathloss_exp / 2.0)
    field_radius_sq = torch.sum(aav_positions**2, dim=-1, keepdim=True)
    field_profile = torch.exp(
        -field_radius_sq / (2.0 * model.shadow_width**2)
    )
    attenuation = torch.exp(-model.shadow_strength * field_profile)
    path_gain = path_gain * attenuation
    return torch.log1p(path_gain / SIGMA)


def make_validation_scenario(seed: int, num_users: int = 4) -> Scenario:
    """Create a corridor layout whose direct path crosses the hidden field."""
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    x = 2.7 + 1.5 * torch.rand(
        num_users, generator=generator, device=DEVICE
    )
    y = -2.8 + 5.6 * torch.rand(
        num_users, generator=generator, device=DEVICE
    )
    user_positions = torch.stack([x, y], dim=1)
    tasks = 2.0 + 1.5 * torch.rand(
        num_users, generator=generator, device=DEVICE
    )
    start_positions = torch.tensor([[-4.0, 0.0]], device=DEVICE)
    obstacles = torch.empty((0, 3), device=DEVICE)
    return Scenario(user_positions, tasks, start_positions, obstacles)


def make_action_branches(
    seed: int,
    n_branches: int,
    horizon: int,
    num_users: int,
    num_aavs: int = 1,
) -> list[ActionNoise]:
    """Create one baseline and paired, temporally correlated action branches."""
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    if num_aavs == 1:
        zero_angle = torch.zeros(horizon, device=DEVICE)
        zero_speed = torch.zeros(horizon, device=DEVICE)
        zero_allocation = torch.zeros(
            (horizon, num_users), device=DEVICE
        )
    else:
        zero_angle = torch.zeros((horizon, num_aavs), device=DEVICE)
        zero_speed = torch.zeros((horizon, num_aavs), device=DEVICE)
        zero_allocation = torch.zeros(
            (horizon, num_aavs, num_users), device=DEVICE
        )
    branches = [ActionNoise(zero_angle, zero_speed, zero_allocation)]
    pair_count = math.ceil((n_branches - 1) / 2)
    for pair_index in range(pair_count):
        if pair_index < 2:
            smooth_angle = torch.zeros(horizon, device=DEVICE)
            amplitude = (0.10, 0.25)[pair_index]
            pulse_divisor = (5, 4)[pair_index]
            pulse_steps = min(horizon, max(5, horizon // pulse_divisor))
            smooth_angle[:pulse_steps] = amplitude
            raw_speed = torch.zeros(horizon, device=DEVICE)
            raw_allocation = torch.zeros(
                (horizon, num_users), device=DEVICE
            )
        else:
            raw_angle = torch.randn(
                horizon, generator=generator, device=DEVICE
            )
            kernel = torch.ones(7, device=DEVICE) / 7.0
            smooth_angle = F.conv1d(
                raw_angle.view(1, 1, -1),
                kernel.view(1, 1, -1),
                padding=3,
            ).view(-1)
            smooth_angle = 0.75 * smooth_angle / (
                smooth_angle.std() + 1e-6
            )
            raw_speed = 0.035 * torch.randn(
                horizon, generator=generator, device=DEVICE
            )
            raw_allocation = 0.55 * torch.randn(
                (
                    (horizon, num_users)
                    if num_aavs == 1
                    else (horizon, num_aavs, num_users)
                ),
                generator=generator,
                device=DEVICE,
            )
        if num_aavs > 1:
            if pair_index % 2 == 0:
                fleet_pattern = torch.ones(num_aavs, device=DEVICE)
            else:
                fleet_pattern = torch.linspace(
                    -1.0, 1.0, num_aavs, device=DEVICE
                )
            smooth_angle = smooth_angle.unsqueeze(1) * fleet_pattern
            raw_speed = raw_speed.unsqueeze(1) * torch.flip(
                fleet_pattern, dims=(0,)
            )
            if raw_allocation.ndim == 2:
                raw_allocation = raw_allocation.unsqueeze(1).repeat(
                    1, num_aavs, 1
                )
        for sign in (1.0, -1.0):
            branches.append(
                ActionNoise(
                    sign * smooth_angle,
                    sign * raw_speed,
                    sign * raw_allocation,
                )
            )
            if len(branches) == n_branches:
                return branches
    return branches[:n_branches]


def action_features(
    angle: torch.Tensor,
    speed: torch.Tensor,
    allocation: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.cos(angle).reshape(-1),
            torch.sin(angle).reshape(-1),
            (speed / MAX_STEP).reshape(-1),
            allocation.reshape(-1),
        ]
    )


def rollout_objective(
    policy: FleetPolicy,
    scenario: Scenario,
    model: WorldModelSpec,
    horizon: int,
    noise: ActionNoise,
    keep_trace: bool = False,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Differentiably unroll one candidate action trajectory."""
    positions = scenario.start_positions.clone()
    backlog = scenario.tasks.clone()
    initial_total = torch.sum(scenario.tasks)
    objective = torch.zeros((), device=DEVICE)
    trace: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    empty_obstacles = torch.empty((0, 3), device=DEVICE)

    for step in range(horizon):
        state_positions = positions
        state_backlog = backlog
        angle, speed, allocation = policy(
            positions,
            scenario.user_positions,
            backlog,
            empty_obstacles,
            max_step=MAX_STEP,
        )
        angle = angle + noise.angle[step].reshape_as(angle)
        speed = torch.clamp(
            speed + noise.speed[step].reshape_as(speed), 0.0, MAX_STEP
        )
        allocation_logits = torch.log(allocation.reshape(-1) + 1e-8)
        allocation = F.softmax(
            allocation_logits + noise.allocation[step], dim=0
        ).view_as(allocation)
        positions = advance(positions, angle, speed)
        rates = channel_rates_parametric(
            positions, scenario.user_positions, model
        )
        delivered = (
            allocation * scenario.tasks.numel() * rates * TIME_STEP
        ).sum(dim=0)
        backlog = torch.clamp(backlog - delivered, min=0.0)
        objective = objective + torch.sum(backlog) / initial_total
        boundary = torch.clamp(torch.abs(positions) - 5.0, min=0.0)
        objective = objective + 0.1 * torch.sum(boundary)
        if keep_trace:
            trace.append(
                (
                    state_positions.detach(),
                    state_backlog.detach(),
                    action_features(angle, speed, allocation).detach(),
                )
            )
        if float(torch.sum(backlog).detach()) < 1e-3 * backlog.numel():
            break
    return objective, trace


def deterministic_evaluation(
    policy: FleetPolicy,
    scenario: Scenario,
    model: WorldModelSpec,
    horizon: int,
    noise: ActionNoise | None = None,
) -> dict[str, float | bool | np.ndarray]:
    """Evaluate a policy without exposing the true model to training."""
    if noise is None:
        noise = make_action_branches(
            0, 1, horizon, scenario.tasks.numel()
        )[0]
    positions = scenario.start_positions.clone()
    backlog = scenario.tasks.clone()
    initial_total = float(torch.sum(backlog))
    backlog_area = 0.0
    trajectory = [positions.detach().cpu().numpy().copy()]
    empty_obstacles = torch.empty((0, 3), device=DEVICE)
    completed = False
    steps = horizon
    with torch.no_grad():
        for step in range(horizon):
            angle, speed, allocation = policy(
                positions,
                scenario.user_positions,
                backlog,
                empty_obstacles,
                max_step=MAX_STEP,
            )
            angle = angle + noise.angle[step].reshape_as(angle)
            speed = torch.clamp(
                speed + noise.speed[step].reshape_as(speed), 0.0, MAX_STEP
            )
            logits = torch.log(allocation.reshape(-1) + 1e-8)
            allocation = F.softmax(
                logits + noise.allocation[step], dim=0
            ).view_as(allocation)
            positions = advance(positions, angle, speed)
            rates = channel_rates_parametric(
                positions, scenario.user_positions, model
            )
            delivered = (
                allocation * scenario.tasks.numel() * rates * TIME_STEP
            ).sum(dim=0)
            backlog = torch.clamp(backlog - delivered, min=0.0)
            backlog_area += float(torch.sum(backlog)) / initial_total
            trajectory.append(positions.detach().cpu().numpy().copy())
            if float(torch.sum(backlog)) < 1e-3 * backlog.numel():
                completed = True
                steps = step + 1
                break
    return {
        "steps": float(steps),
        "completed": completed,
        "backlog_area": backlog_area,
        "remaining": float(torch.sum(backlog)),
        "trajectory": np.concatenate(trajectory, axis=0),
    }


def robust_branch_scores(
    policy: FleetPolicy,
    scenario: Scenario,
    branches: list[ActionNoise],
    horizon: int,
    models: tuple[WorldModelSpec, ...],
    need_traces: bool,
) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    branch_scores: list[torch.Tensor] = []
    branch_traces: list[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = []
    central_index = len(models) // 2
    for branch in branches:
        model_losses: list[torch.Tensor] = []
        selected_trace: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        for model_index, model in enumerate(models):
            loss, trace = rollout_objective(
                policy,
                scenario,
                model,
                horizon,
                branch,
                keep_trace=need_traces and model_index == central_index,
            )
            model_losses.append(loss)
            if trace:
                selected_trace = trace
        stacked = torch.stack(model_losses)
        score = torch.mean(stacked)
        if len(models) > 1:
            score = score + ROBUSTNESS_WEIGHT * torch.std(
                stacked, unbiased=False
            )
        branch_scores.append(score)
        branch_traces.append(selected_trace)
    return torch.stack(branch_scores), branch_traces


def trace_distillation_loss(
    policy: FleetPolicy,
    scenario: Scenario,
    trace: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    positive: bool,
) -> torch.Tensor:
    if not trace:
        return torch.zeros((), device=DEVICE)
    empty_obstacles = torch.empty((0, 3), device=DEVICE)
    distances = []
    stride = max(1, len(trace) // 24)
    for positions, backlog, target in trace[::stride]:
        angle, speed, allocation = policy(
            positions,
            scenario.user_positions,
            backlog,
            empty_obstacles,
            max_step=MAX_STEP,
        )
        prediction = action_features(angle, speed, allocation)
        distances.append(torch.mean((prediction - target) ** 2))
    mean_distance = torch.mean(torch.stack(distances))
    if positive:
        return mean_distance
    return F.relu(DISTANCE_MARGIN - mean_distance)


def policy_vector(policy: FleetPolicy) -> np.ndarray:
    return torch.cat(
        [parameter.detach().flatten().cpu() for parameter in policy.parameters()]
    ).numpy()


def train_method(
    method: str,
    scenario: Scenario,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, np.ndarray | float | FleetPolicy]:
    torch.manual_seed(seed)
    policy = FleetPolicy(
        scenario.tasks.numel(), num_aavs=1, num_obstacles=0
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=1e-3, weight_decay=0.0
    )
    training_objective = np.zeros(config.train_episodes)
    update_norm = np.zeros(config.train_episodes)
    positive_counts = np.zeros(config.train_episodes)
    negative_counts = np.zeros(config.train_episodes)
    ensemble_labels: list[int] = []
    nominal_labels: list[int] = []
    true_labels: list[int] = []
    checkpoint_episodes: list[int] = []
    checkpoint_steps: list[float] = []
    checkpoint_backlog: list[float] = []
    start_time = time.perf_counter()

    for episode in range(config.train_episodes):
        branches = make_action_branches(
            seed * 100_000 + episode,
            config.n_branches,
            config.train_horizon,
            scenario.tasks.numel(),
        )
        if method == "Single trial":
            objective, _ = rollout_objective(
                policy,
                scenario,
                NOMINAL_MODEL,
                config.train_horizon,
                branches[0],
            )
        elif method == "Ensemble only":
            scores, _ = robust_branch_scores(
                policy,
                scenario,
                [branches[0]],
                config.train_horizon,
                ENSEMBLE_MODELS,
                need_traces=False,
            )
            objective = scores[0]
        elif method == "Random multi-trial":
            scores, _ = robust_branch_scores(
                policy,
                scenario,
                branches,
                config.train_horizon,
                (NOMINAL_MODEL,),
                need_traces=False,
            )
            objective = torch.mean(scores)
        elif method == "Counterfactual P/N":
            # The ensemble acts as a counterfactual teacher. Its labels and
            # traces are detached; the stable nominal pathwise objective stays
            # the main task gradient.
            with torch.no_grad():
                scores, traces = robust_branch_scores(
                    policy,
                    scenario,
                    branches,
                    config.train_horizon,
                    ENSEMBLE_MODELS,
                    need_traces=True,
                )
            detached_scores = scores.detach()
            baseline = detached_scores[0]
            threshold = max(0.005, 0.0005 * abs(float(baseline)))
            candidate_scores = detached_scores[1:]
            positive_mask = candidate_scores < baseline - threshold
            negative_mask = candidate_scores > baseline + threshold
            positive_counts[episode] = float(torch.sum(positive_mask))
            negative_counts[episode] = float(torch.sum(negative_mask))
            has_positive = bool(torch.any(positive_mask))
            best_index = int(torch.argmin(detached_scores))
            worst_index = int(torch.argmax(detached_scores))
            objective, _ = rollout_objective(
                policy,
                scenario,
                NOMINAL_MODEL,
                config.train_horizon,
                branches[0],
            )
            if has_positive and best_index != 0 and bool(
                detached_scores[best_index] < baseline - threshold
            ):
                predicted_gain = baseline - detached_scores[best_index]
                confidence_weight = torch.clamp(
                    predicted_gain / 0.50, min=0.0, max=1.0
                )
                objective = (
                    objective
                    + POSITIVE_WEIGHT
                    * confidence_weight
                    * trace_distillation_loss(
                        policy, scenario, traces[best_index], positive=True
                    )
                )
            if has_positive and worst_index != 0 and bool(
                detached_scores[worst_index] > baseline + threshold
            ):
                objective = objective + NEGATIVE_WEIGHT * trace_distillation_loss(
                    policy, scenario, traces[worst_index], positive=False
                )

            if episode % max(1, config.checkpoint_every // 2) == 0:
                with torch.no_grad():
                    nominal_scores, _ = robust_branch_scores(
                        policy,
                        scenario,
                        branches,
                        config.train_horizon,
                        (NOMINAL_MODEL,),
                        need_traces=False,
                    )
                    true_scores, _ = robust_branch_scores(
                        policy,
                        scenario,
                        branches,
                        config.train_horizon,
                        (TRUE_MODEL,),
                        need_traces=False,
                    )
                nominal_delta = nominal_scores[0] - nominal_scores[1:]
                ensemble_delta = detached_scores[0] - detached_scores[1:]
                true_delta = true_scores[0] - true_scores[1:]
                audit_threshold = threshold
                for nominal_value, ensemble_value, true_value in zip(
                    nominal_delta, ensemble_delta, true_delta
                ):
                    nominal_labels.append(
                        1
                        if float(nominal_value) > audit_threshold
                        else -1
                        if float(nominal_value) < -audit_threshold
                        else 0
                    )
                    ensemble_labels.append(
                        1
                        if float(ensemble_value) > audit_threshold
                        else -1
                        if float(ensemble_value) < -audit_threshold
                        else 0
                    )
                    true_labels.append(
                        1
                        if float(true_value) > audit_threshold
                        else -1
                        if float(true_value) < -audit_threshold
                        else 0
                    )
        else:
            raise ValueError(f"Unknown method: {method}")

        optimizer.zero_grad()
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=1.0
        )
        optimizer.step()
        objective_value = float(objective.detach())
        if not np.isfinite(objective_value):
            raise FloatingPointError(
                f"Non-finite objective for {method}, episode {episode}."
            )
        training_objective[episode] = objective_value
        update_norm[episode] = float(grad_norm)

        if (
            episode % config.checkpoint_every == 0
            or episode == config.train_episodes - 1
        ):
            result = deterministic_evaluation(
                policy,
                scenario,
                TRUE_MODEL,
                config.eval_horizon,
            )
            checkpoint_episodes.append(episode + 1)
            checkpoint_steps.append(float(result["steps"]))
            checkpoint_backlog.append(float(result["backlog_area"]))

    wall_time = time.perf_counter() - start_time
    final = deterministic_evaluation(
        policy, scenario, TRUE_MODEL, config.eval_horizon
    )
    return {
        "policy": policy,
        "training_objective": training_objective,
        "update_norm": update_norm,
        "positive_counts": positive_counts,
        "negative_counts": negative_counts,
        "nominal_labels": np.asarray(nominal_labels, dtype=np.int8),
        "ensemble_labels": np.asarray(ensemble_labels, dtype=np.int8),
        "true_labels": np.asarray(true_labels, dtype=np.int8),
        "checkpoint_episodes": np.asarray(checkpoint_episodes),
        "checkpoint_steps": np.asarray(checkpoint_steps),
        "checkpoint_backlog": np.asarray(checkpoint_backlog),
        "final_steps": float(final["steps"]),
        "final_completed": float(final["completed"]),
        "final_backlog_area": float(final["backlog_area"]),
        "final_remaining": float(final["remaining"]),
        "trajectory": np.asarray(final["trajectory"]),
        "wall_time": wall_time,
    }


def label_accuracy(
    predicted: np.ndarray, truth: np.ndarray
) -> tuple[float, int]:
    mask = predicted != 0
    if not np.any(mask):
        return float("nan"), 0
    return float(np.mean(predicted[mask] == truth[mask])), int(np.sum(mask))


def save_table(
    path: Path,
    steps: np.ndarray,
    success: np.ndarray,
    backlog: np.ndarray,
    wall_times: np.ndarray,
    nominal_accuracy: float,
    ensemble_accuracy: float,
) -> None:
    med_steps = np.median(steps, axis=0)
    success_rates = 100.0 * np.mean(success, axis=0)
    med_backlog = np.median(backlog, axis=0)
    med_times = np.median(wall_times, axis=0)
    best_steps = np.min(med_steps)
    best_success = np.max(success_rates)
    best_backlog = np.min(med_backlog)
    baseline_index = METHODS.index("Random multi-trial")
    ours_index = METHODS.index("Counterfactual P/N")
    relative_gain = (
        (med_steps[baseline_index] - med_steps[ours_index])
        / max(med_steps[baseline_index], 1e-9)
        * 100.0
    )
    absolute_gain = med_steps[baseline_index] - med_steps[ours_index]
    success_gain = (
        success_rates[ours_index] - success_rates[baseline_index]
    )
    backlog_relative_gain = (
        (med_backlog[baseline_index] - med_backlog[ours_index])
        / max(med_backlog[baseline_index], 1e-9)
        * 100.0
    )
    backlog_absolute_gain = (
        med_backlog[baseline_index] - med_backlog[ours_index]
    )

    lines = [
        "| Method | True-model completion slots $\\downarrow$ | Success $\\uparrow$ | "
        "Backlog area $\\downarrow$ | Training time (s) $\\downarrow$ |",
        "|---|---:|---:|---:|---:|",
    ]
    for index, method in enumerate(METHODS):
        values = [
            f"{med_steps[index]:.1f}",
            f"{success_rates[index]:.1f}%",
            f"{med_backlog[index]:.2f}",
            f"{med_times[index]:.2f}",
        ]
        best_flags = (
            np.isclose(med_steps[index], best_steps),
            np.isclose(success_rates[index], best_success),
            np.isclose(med_backlog[index], best_backlog),
            False,
        )
        for value_index, is_best in enumerate(best_flags):
            if is_best:
                values[value_index] = f"**{values[value_index]}**"
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "| **Gain over random multi-trial** | "
            f"**{relative_gain:+.1f}% ({absolute_gain:+.1f} slots)** | "
            f"**{success_gain:+.1f} pp** | "
            f"**{backlog_relative_gain:+.1f}% "
            f"({backlog_absolute_gain:+.2f})** | -- |",
            "",
            f"Nominal-WM counterfactual-label accuracy: **{100*nominal_accuracy:.1f}%**.",
            "",
            f"Ensemble-robust counterfactual-label accuracy: "
            f"**{100*ensemble_accuracy:.1f}%**.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(
    figure_dir: Path,
    steps: np.ndarray,
    convergence_backlog: np.ndarray,
    checkpoint_episodes: np.ndarray,
    nominal_accuracy: float,
    ensemble_accuracy: float,
    nominal_labels: np.ndarray,
    ensemble_labels: np.ndarray,
    true_labels: np.ndarray,
) -> None:
    rng = np.random.default_rng(20260730)
    fig, ax = plt.subplots(figsize=(6, 6))
    positions = np.arange(1, len(METHODS) + 1)
    box = ax.boxplot(
        [steps[:, index] for index in range(len(METHODS))],
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2.0},
    )
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.28)
        patch.set_edgecolor(METHOD_COLORS[method])
        patch.set_linewidth(1.5)
    for index, method in enumerate(METHODS):
        jitter = rng.normal(positions[index], 0.045, size=steps.shape[0])
        ax.scatter(
            jitter,
            steps[:, index],
            s=38,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label=method,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["Single", "Ensemble", "Random\nmulti", "Counterfactual\nP/N"],
        fontsize=12,
    )
    ax.set_ylabel("Completion time (slots)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.05)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"multitrial_wm_completion.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for method_index, method in enumerate(METHODS):
        median = np.median(convergence_backlog[:, method_index], axis=0)
        lower = np.percentile(
            convergence_backlog[:, method_index], 25, axis=0
        )
        upper = np.percentile(
            convergence_backlog[:, method_index], 75, axis=0
        )
        ax.plot(
            checkpoint_episodes,
            median,
            color=METHOD_COLORS[method],
            linewidth=2.2,
            marker="o",
            markersize=4.5,
            label=method,
        )
        ax.fill_between(
            checkpoint_episodes,
            lower,
            upper,
            color=METHOD_COLORS[method],
            alpha=0.14,
            linewidth=0,
        )
    ax.set_xlabel("Policy-update episode", fontsize=14)
    ax.set_ylabel("Backlog area", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"multitrial_wm_convergence.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    values = 100 * np.asarray([nominal_accuracy, ensemble_accuracy])
    labels = ["Nominal-WM\nlabels", "Ensemble-robust\nlabels"]
    colors = [METHOD_COLORS["Single trial"], METHOD_COLORS["Counterfactual P/N"]]
    bars = ax.bar(
        labels,
        values,
        color=colors,
        alpha=0.82,
        edgecolor="black",
        linewidth=0.8,
        width=0.58,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
        )
    ax.axhline(50, color="black", linestyle=":", linewidth=1.0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("P/N label accuracy (%)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"multitrial_wm_label_accuracy.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)

    nominal_correct = np.cumsum(
        (nominal_labels != 0) & (nominal_labels == true_labels)
    )
    ensemble_correct = np.cumsum(
        (ensemble_labels != 0) & (ensemble_labels == true_labels)
    )
    candidate_index = np.arange(1, true_labels.size + 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        candidate_index,
        nominal_correct,
        color=METHOD_COLORS["Single trial"],
        linewidth=2.2,
        label="Nominal-WM labels",
    )
    ax.plot(
        candidate_index,
        ensemble_correct,
        color=METHOD_COLORS["Counterfactual P/N"],
        linewidth=2.4,
        label="Ensemble-robust labels",
    )
    ax.fill_between(
        candidate_index,
        nominal_correct,
        ensemble_correct,
        where=ensemble_correct >= nominal_correct,
        color=METHOD_COLORS["Counterfactual P/N"],
        alpha=0.12,
    )
    ax.set_xlabel("Audited action branch", fontsize=14)
    ax.set_ylabel("Cumulative correct P/N labels", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.margins(x=0.01, y=0.03)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"multitrial_wm_label_accumulation.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def get_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        return ExperimentConfig(1, 2, 20, 40, 3, 1, "_smoke")
    if args.validate:
        return ExperimentConfig(2, 20, 50, 180, 5, 5, "_validate")
    return ExperimentConfig(8, 25, 55, 220, 5, 5, "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.validate:
        parser.error("--smoke and --validate are mutually exclusive")
    config = get_config(args)

    root = Path(__file__).resolve().parent
    result_dir = root / "results_extended" / "multitrial_wm"
    raw_dir = result_dir / "raw"
    figure_dir = root / "paper_assets" / "figures"
    table_dir = root / "paper_assets" / "tables"
    for directory in (result_dir, raw_dir, figure_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shape = (config.n_trials, len(METHODS))
    final_steps = np.zeros(shape)
    final_success = np.zeros(shape)
    final_backlog = np.zeros(shape)
    final_remaining = np.zeros(shape)
    wall_times = np.zeros(shape)
    objectives = np.zeros(
        (config.n_trials, len(METHODS), config.train_episodes)
    )
    update_norms = np.zeros_like(objectives)
    positive_counts = np.zeros_like(objectives)
    negative_counts = np.zeros_like(objectives)
    checkpoint_count = math.ceil(
        (config.train_episodes - 1) / config.checkpoint_every
    ) + 1
    convergence_steps = np.zeros(
        (config.n_trials, len(METHODS), checkpoint_count)
    )
    convergence_backlog = np.zeros_like(convergence_steps)
    trajectories = np.empty(shape, dtype=object)
    policy_vectors = np.empty(shape, dtype=object)
    all_nominal_labels: list[np.ndarray] = []
    all_ensemble_labels: list[np.ndarray] = []
    all_true_labels: list[np.ndarray] = []
    checkpoint_episodes: np.ndarray | None = None

    for trial in range(config.n_trials):
        scenario_seed = 73000 + trial
        initialization_seed = 91000 + trial
        scenario = make_validation_scenario(scenario_seed, num_users=4)
        raw_payload: dict[str, np.ndarray] = {
            "scenario_seed": np.asarray(scenario_seed),
            "initialization_seed": np.asarray(initialization_seed),
            "user_positions": scenario.user_positions.detach().cpu().numpy(),
            "user_tasks": scenario.tasks.detach().cpu().numpy(),
            "start_positions": scenario.start_positions.detach().cpu().numpy(),
        }
        for method_index, method in enumerate(METHODS):
            result = train_method(
                method, scenario, config, initialization_seed
            )
            policy = result.pop("policy")
            final_steps[trial, method_index] = result["final_steps"]
            final_success[trial, method_index] = result["final_completed"]
            final_backlog[trial, method_index] = result["final_backlog_area"]
            final_remaining[trial, method_index] = result["final_remaining"]
            wall_times[trial, method_index] = result["wall_time"]
            objectives[trial, method_index] = result["training_objective"]
            update_norms[trial, method_index] = result["update_norm"]
            positive_counts[trial, method_index] = result["positive_counts"]
            negative_counts[trial, method_index] = result["negative_counts"]
            convergence_steps[trial, method_index] = result[
                "checkpoint_steps"
            ]
            convergence_backlog[trial, method_index] = result[
                "checkpoint_backlog"
            ]
            trajectories[trial, method_index] = result["trajectory"]
            policy_vectors[trial, method_index] = policy_vector(policy)
            checkpoint_episodes = result["checkpoint_episodes"]
            if method == "Counterfactual P/N":
                all_nominal_labels.append(result["nominal_labels"])
                all_ensemble_labels.append(result["ensemble_labels"])
                all_true_labels.append(result["true_labels"])
            token = method.lower().replace(" ", "_").replace("/", "n")
            for key, value in result.items():
                raw_payload[f"{token}_{key}"] = np.asarray(value)
            raw_payload[f"{token}_policy_vector"] = policy_vectors[
                trial, method_index
            ]
            print(
                f"trial {trial + 1}/{config.n_trials}, {method}: "
                f"steps={final_steps[trial, method_index]:.0f}, "
                f"success={bool(final_success[trial, method_index])}, "
                f"time={wall_times[trial, method_index]:.2f}s",
                flush=True,
            )
        raw_payload["methods"] = np.asarray(METHODS)
        raw_payload["metadata_json"] = np.asarray(
            json.dumps(
                {
                    **asdict(config),
                    "nominal_model": asdict(NOMINAL_MODEL),
                    "ensemble_models": [
                        asdict(model) for model in ENSEMBLE_MODELS
                    ],
                    "true_model": asdict(TRUE_MODEL),
                    "device": str(DEVICE),
                }
            )
        )
        raw_path = (
            raw_dir
            / f"multitrial_wm{config.tag}_trial{trial + 1:02d}.npz"
        )
        np.savez_compressed(raw_path, **raw_payload)

    nominal_labels = np.concatenate(all_nominal_labels)
    ensemble_labels = np.concatenate(all_ensemble_labels)
    true_labels = np.concatenate(all_true_labels)
    nominal_accuracy, nominal_count = label_accuracy(
        nominal_labels, true_labels
    )
    ensemble_accuracy, ensemble_count = label_accuracy(
        ensemble_labels, true_labels
    )
    dense_policy_vectors = np.stack(
        [
            np.stack(
                [policy_vectors[trial, method] for method in range(len(METHODS))]
            )
            for trial in range(config.n_trials)
        ]
    )
    dense_trajectories = np.empty_like(trajectories)
    dense_trajectories[:] = trajectories
    aggregate = {
        "methods": np.asarray(METHODS),
        "final_steps": final_steps,
        "final_success": final_success,
        "final_backlog_area": final_backlog,
        "final_remaining": final_remaining,
        "training_objectives": objectives,
        "update_norms": update_norms,
        "positive_counts": positive_counts,
        "negative_counts": negative_counts,
        "convergence_steps": convergence_steps,
        "convergence_backlog_area": convergence_backlog,
        "checkpoint_episodes": checkpoint_episodes,
        "wall_times": wall_times,
        "policy_vectors": dense_policy_vectors,
        "trajectories": dense_trajectories,
        "nominal_labels": nominal_labels,
        "ensemble_labels": ensemble_labels,
        "true_labels": true_labels,
        "nominal_label_accuracy": np.asarray(nominal_accuracy),
        "ensemble_label_accuracy": np.asarray(ensemble_accuracy),
        "nominal_label_count": np.asarray(nominal_count),
        "ensemble_label_count": np.asarray(ensemble_count),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    **asdict(config),
                    "nominal_model": asdict(NOMINAL_MODEL),
                    "ensemble_models": [
                        asdict(model) for model in ENSEMBLE_MODELS
                    ],
                    "true_model": asdict(TRUE_MODEL),
                    "robustness_weight": ROBUSTNESS_WEIGHT,
                    "positive_weight": POSITIVE_WEIGHT,
                    "negative_weight": NEGATIVE_WEIGHT,
                    "distance_margin": DISTANCE_MARGIN,
                    "device": str(DEVICE),
                }
            )
        ),
    }
    output_path = result_dir / f"multitrial_wm_validation{config.tag}.npz"
    np.savez_compressed(output_path, **aggregate)
    save_table(
        table_dir / f"table_multitrial_wm_validation{config.tag}.md",
        final_steps,
        final_success,
        final_backlog,
        wall_times,
        nominal_accuracy,
        ensemble_accuracy,
    )
    plot_results(
        figure_dir,
        final_steps,
        convergence_backlog,
        checkpoint_episodes,
        nominal_accuracy,
        ensemble_accuracy,
        nominal_labels,
        ensemble_labels,
        true_labels,
    )
    print(f"Saved {output_path}")
    print(
        f"Label agreement: nominal={100*nominal_accuracy:.1f}% "
        f"(n={nominal_count}), ensemble={100*ensemble_accuracy:.1f}% "
        f"(n={ensemble_count})"
    )
    if args.smoke:
        print("Smoke test PASSED")
    elif args.validate:
        print("Validation run PASSED")


if __name__ == "__main__":
    main()

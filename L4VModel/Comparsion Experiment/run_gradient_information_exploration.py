"""Incremental validation of gradient-informative world-model exploration.

The experiment adds the proposed mechanisms in a fixed order:

1. action-diverse counterfactual branches;
2. gradient-information branch selection;
3. confidence-weighted local positive distillation and local rejection;
4. state-dependent trust regions;
5. common random numbers for paired branch comparisons;
6. limited high-fidelity queries and online parametric calibration; and
7. a physics core with bootstrapped learned residual models.

Every variant uses paired scenarios, identical policy initialization, and the
same number of policy updates.  The high-fidelity world model is available to
the last two variants through at most one ambiguous-branch query per update.
For all variants it is also used after the update for retrospective auditing
and final evaluation.  Audit calls never affect variants without query access.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_multitrial_wm_exploration import (
    ActionNoise,
    MAX_STEP,
    SIGMA,
    TIME_STEP,
    WorldModelSpec,
    action_features,
    make_action_branches,
    make_validation_scenario,
)
from run_stress_tests import (
    DEVICE,
    FleetPolicy,
    Scenario,
    advance,
    constraint_penalty,
)


CHANNEL_NOISE_STD = 0.16
ROBUSTNESS_WEIGHT = 0.15
POSITIVE_WEIGHT = 45.0
NEGATIVE_REJECTION_WEIGHT = 2.0
LOCAL_REJECTION_MARGIN = 0.028
MAX_QUERY_STEPS = 25
OBSTACLE_WEIGHT = 25.0

NOMINAL_SPEC = WorldModelSpec(2.0, 0.45, 0.80)
MANUAL_SPECS = (
    WorldModelSpec(2.0, 1.00, 1.00),
    WorldModelSpec(2.0, 3.00, 1.40),
    WorldModelSpec(2.0, 5.00, 1.80),
)


@dataclass(frozen=True)
class Variant:
    name: str
    gradient_information: bool
    local_pn: bool
    trust_region: bool
    common_random_numbers: bool
    high_fidelity_query: bool
    learned_residual: bool
    selection_strategy: str = ""
    ucb_beta: float = 1.0
    information_weighted_update: bool = False
    update_aligned_descriptor: bool = False
    reliable_useful_weighting: bool = False
    adaptive_selection: bool = False
    adaptive_min_gain: float = 0.18
    multi_horizon_descriptor: bool = False
    task_aware_query: bool = False
    balanced_conflict_safe_update: bool = False
    confidence_scaled_trial_update: bool = False
    force_query: bool = False
    trial_gradient_weight: float = 0.75
    adaptive_pool: bool = False
    minimum_pool_branches: int = 3
    information_gated_trial_update: bool = False
    minimum_trial_reliability: float = 0.15
    trial_weight_half_life: float = 0.0
    wm_checkpoint_selection: bool = False
    task_utility_weight: float = 0.75
    lcb_beta: float = 1.0
    wm_safe_line_search: bool = False
    line_search_margin: float = 0.0
    persistent_crn: bool = False
    hf_checkpoint_selection: bool = False
    line_search_noise_repeats: int = 1
    # --- residual calibration (RC-GIB upgrade) ---
    use_residual_calibration: bool = False
    calib_delta: float = 0.10
    calib_max_samples: int = 50


VARIANTS = (
    Variant("Action diversity", False, False, False, False, False, False),
    Variant("+ Gradient information", True, False, False, False, False, False),
    Variant("+ Local P/N", True, True, False, False, False, False),
    Variant("+ Trust region", True, True, True, False, False, False),
    Variant("+ CRN", True, True, True, True, False, False),
    Variant("+ HF calibration", True, True, True, True, True, False),
    Variant("+ Learned residual", True, True, True, True, True, True),
)

METHOD_COLORS = {
    "Action diversity": "#7f7f7f",
    "+ Gradient information": "#1f77b4",
    "+ Local P/N": "#17becf",
    "+ Trust region": "#ff7f0e",
    "+ CRN": "#9467bd",
    "+ HF calibration": "#8c564b",
    "+ Learned residual": "#d62728",
}

SHORT_LABELS = (
    "Action\ndiversity",
    "+ Gradient\ninformation",
    "+ Local\nP/N",
    "+ Trust\nregion",
    "+ CRN",
    "+ HF\ncalibration",
    "+ Learned\nresidual",
)


@dataclass(frozen=True)
class ExperimentConfig:
    n_trials: int
    train_episodes: int
    train_horizon: int
    eval_horizon: int
    pool_branches: int
    selected_branches: int
    checkpoint_every: int
    tag: str


@dataclass
class CalibrationBuffer:
    aav_xy: list[torch.Tensor]
    user_xy: list[torch.Tensor]
    target_rate: list[torch.Tensor]

    @classmethod
    def empty(cls) -> "CalibrationBuffer":
        return cls([], [], [])

    def add(
        self,
        aav_xy: torch.Tensor,
        user_xy: torch.Tensor,
        target_rate: torch.Tensor,
    ) -> None:
        self.aav_xy.append(aav_xy.detach().cpu())
        self.user_xy.append(user_xy.detach().cpu())
        self.target_rate.append(target_rate.detach().cpu())

    def __len__(self) -> int:
        return sum(int(item.shape[0]) for item in self.target_rate)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.target_rate:
            empty_xy = torch.empty((0, 2), device=DEVICE)
            empty_rate = torch.empty((0,), device=DEVICE)
            return empty_xy, empty_xy.clone(), empty_rate
        return (
            torch.cat(self.aav_xy).to(DEVICE),
            torch.cat(self.user_xy).to(DEVICE),
            torch.cat(self.target_rate).to(DEVICE),
        )


class CalibrationErrorCache:
    """Rolling buffer of WM prediction errors for residual calibration.

    After each high-fidelity query the caller records the signed error

        e_b = ΔJ_b^true − ΔJ_b^WM,

    where ΔJ_b is the task *improvement* (positive = branch beats baseline).
    The conservative correction

        q_{1−δ} = −np.percentile(errors, δ·100)

    is subtracted from the task LCB, protecting against the case where
    every ensemble member is wrong in the same direction (systematic
    over-prediction).  Pure variance-based LCB bounds offer no defence
    against that failure mode; this calibration term does.
    """

    def __init__(self, max_samples: int = 50) -> None:
        self.max_samples = max_samples
        self._errors: list[float] = []

    def record(self, true_improvement: float, wm_improvement: float) -> None:
        """Append one signed error from a completed high-fidelity query."""
        self._errors.append(true_improvement - wm_improvement)
        if len(self._errors) > self.max_samples:
            self._errors.pop(0)

    def correction(self, delta: float = 0.10) -> float:
        """Conservative LCB correction q_{1−delta} = −quantile(errors, delta).

        Returns a positive number when the WM tends to over-predict task
        improvement (errors are mostly negative), pushing the calibrated LCB
        downward so that only branches with genuine predicted gain survive the
        relu gate.  Returns zero while fewer than three observations have
        been collected so that the selector behaves identically to TC-GIB
        during the warm-up phase.
        """
        if len(self._errors) < 3:
            return 0.0
        # δ-th percentile of signed errors is negative when WM over-predicts;
        # its negation is the positive correction to subtract from the LCB.
        return float(-np.percentile(self._errors, delta * 100.0))

    def __len__(self) -> int:
        return len(self._errors)


def _pairwise_geometry(
    aav_positions: torch.Tensor,
    user_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_aavs = aav_positions.shape[0]
    num_users = user_positions.shape[0]
    aav_pairs = aav_positions.repeat_interleave(num_users, dim=0)
    user_pairs = user_positions.repeat(num_aavs, 1)
    return aav_pairs, user_pairs


def _parametric_pair_rates(
    aav_xy: torch.Tensor,
    user_xy: torch.Tensor,
    pathloss_exp: torch.Tensor | float,
    shadow_strength: torch.Tensor | float,
    shadow_width: torch.Tensor | float,
) -> torch.Tensor:
    horizontal_sq = torch.sum((aav_xy - user_xy) ** 2, dim=-1)
    horizontal_sq = torch.clamp(horizontal_sq, min=0.1)
    path_gain = horizontal_sq.pow(-pathloss_exp / 2.0)
    radius_sq = torch.sum(aav_xy**2, dim=-1)
    field_profile = torch.exp(
        -radius_sq / (2.0 * torch.as_tensor(shadow_width, device=DEVICE) ** 2)
    )
    attenuation = torch.exp(
        -torch.as_tensor(shadow_strength, device=DEVICE) * field_profile
    )
    return torch.log1p(path_gain * attenuation / SIGMA)


class RateWorldModel(nn.Module):
    def pair_rates(
        self, aav_xy: torch.Tensor, user_xy: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def rates(
        self,
        aav_positions: torch.Tensor,
        user_positions: torch.Tensor,
    ) -> torch.Tensor:
        aav_pairs, user_pairs = _pairwise_geometry(
            aav_positions, user_positions
        )
        return self.pair_rates(aav_pairs, user_pairs).reshape(
            aav_positions.shape[0], user_positions.shape[0]
        )

    def rates_at_step(
        self,
        aav_positions: torch.Tensor,
        user_positions: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Evaluate a stationary model or allow a subclass to model drift."""
        del step
        return self.rates(aav_positions, user_positions)


class FixedParametricWorldModel(RateWorldModel):
    def __init__(self, spec: WorldModelSpec):
        super().__init__()
        self.spec = spec

    def pair_rates(
        self, aav_xy: torch.Tensor, user_xy: torch.Tensor
    ) -> torch.Tensor:
        return _parametric_pair_rates(
            aav_xy,
            user_xy,
            self.spec.pathloss_exp,
            self.spec.shadow_strength,
            self.spec.shadow_width,
        )


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(value, 1e-4)))


class CalibratedParametricWorldModel(RateWorldModel):
    """Trainable radial model used before the learned-residual replacement."""

    def __init__(self, spec: WorldModelSpec):
        super().__init__()
        self.pathloss_exp = spec.pathloss_exp
        self.raw_strength = nn.Parameter(
            torch.tensor(_inverse_softplus(spec.shadow_strength), device=DEVICE)
        )
        self.raw_width = nn.Parameter(
            torch.tensor(_inverse_softplus(spec.shadow_width), device=DEVICE)
        )

    def pair_rates(
        self, aav_xy: torch.Tensor, user_xy: torch.Tensor
    ) -> torch.Tensor:
        strength = F.softplus(self.raw_strength) + 1e-3
        width = F.softplus(self.raw_width) + 0.15
        return _parametric_pair_rates(
            aav_xy, user_xy, self.pathloss_exp, strength, width
        )


class ResidualWorldModel(RateWorldModel):
    """Nominal physics core plus a smooth, bootstrapped RBF rate residual."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        center_x = torch.linspace(-3.6, 3.6, 5, device=DEVICE)
        center_y = torch.linspace(-2.4, 2.4, 3, device=DEVICE)
        mesh_x, mesh_y = torch.meshgrid(
            center_x, center_y, indexing="ij"
        )
        self.register_buffer(
            "centers",
            torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=1),
        )
        self.width = 1.35
        self.linear = nn.Linear(2 * self.centers.shape[0] + 1, 1).to(DEVICE)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.006)
        nn.init.zeros_(self.linear.bias)

    def features(
        self,
        aav_xy: torch.Tensor,
        base_rate: torch.Tensor,
    ) -> torch.Tensor:
        squared_distance = torch.sum(
            (aav_xy.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2,
            dim=-1,
        )
        rbf = torch.exp(
            -squared_distance / (2.0 * self.width**2)
        )
        return torch.cat(
            [
                rbf,
                rbf * (base_rate.unsqueeze(-1) / 3.0),
                base_rate.unsqueeze(-1) / 3.0,
            ],
            dim=-1,
        )

    def pair_rates(
        self, aav_xy: torch.Tensor, user_xy: torch.Tensor
    ) -> torch.Tensor:
        base_rate = _parametric_pair_rates(
            aav_xy,
            user_xy,
            NOMINAL_SPEC.pathloss_exp,
            NOMINAL_SPEC.shadow_strength,
            NOMINAL_SPEC.shadow_width,
        )
        residual = 1.25 * torch.tanh(
            self.linear(self.features(aav_xy, base_rate)).squeeze(-1)
        )
        return torch.clamp(base_rate + residual, min=1e-5)


class HighFidelityWorldModel(RateWorldModel):
    """Out-of-family anisotropic attenuation used as the query/evaluation WM."""

    def pair_rates(
        self, aav_xy: torch.Tensor, user_xy: torch.Tensor
    ) -> torch.Tensor:
        horizontal_sq = torch.sum((aav_xy - user_xy) ** 2, dim=-1)
        horizontal_sq = torch.clamp(horizontal_sq, min=0.1)
        path_gain = horizontal_sq.pow(-1.0)

        centered = aav_xy - torch.tensor([0.15, -0.10], device=DEVICE)
        angle = 0.45
        rotated_x = math.cos(angle) * centered[:, 0] + math.sin(angle) * centered[:, 1]
        rotated_y = -math.sin(angle) * centered[:, 0] + math.cos(angle) * centered[:, 1]
        primary = torch.exp(
            -0.5 * ((rotated_x / 0.72) ** 2 + (rotated_y / 1.55) ** 2)
        )
        secondary_center = aav_xy - torch.tensor([-0.85, 1.10], device=DEVICE)
        secondary = torch.exp(
            -0.5
            * (
                (secondary_center[:, 0] / 0.55) ** 2
                + (secondary_center[:, 1] / 0.75) ** 2
            )
        )
        attenuation = torch.exp(-4.0 * primary - 1.6 * secondary)
        return torch.log1p(path_gain * attenuation / SIGMA)


NOMINAL_MODEL = FixedParametricWorldModel(NOMINAL_SPEC).to(DEVICE)
TRUE_MODEL = HighFidelityWorldModel().to(DEVICE)


def build_ensemble(variant: Variant, seed: int) -> list[RateWorldModel]:
    if variant.learned_residual:
        return [
            ResidualWorldModel(seed + 1009 * index).to(DEVICE)
            for index in range(len(MANUAL_SPECS))
        ]
    if variant.high_fidelity_query:
        return [
            CalibratedParametricWorldModel(spec).to(DEVICE)
            for spec in MANUAL_SPECS
        ]
    return [
        FixedParametricWorldModel(spec).to(DEVICE) for spec in MANUAL_SPECS
    ]


def make_channel_noise(
    seed: int,
    horizon: int,
    num_aavs: int,
    num_users: int,
) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    return torch.randn(
        (horizon, num_aavs, num_users),
        generator=generator,
        device=DEVICE,
    )


def safety_projected_advance(
    positions: torch.Tensor,
    angles: torch.Tensor,
    speeds: torch.Tensor,
    obstacles: torch.Tensor,
    obstacle_margin: float = 0.12,
    min_separation: float = 0.50,
) -> torch.Tensor:
    """Apply a differentiable known-geometry safety projection."""
    proposed = advance(positions, angles, speeds)
    if obstacles.numel():
        for obstacle in obstacles:
            delta = proposed - obstacle[:2]
            distance = torch.linalg.vector_norm(
                delta, dim=1, keepdim=True
            )
            safe_radius = obstacle[2] + obstacle_margin
            penetration = torch.relu(safe_radius - distance)
            direction = delta / torch.clamp(distance, min=1e-6)
            proposed = proposed + penetration * direction
    if proposed.shape[0] > 1:
        correction = torch.zeros_like(proposed)
        for first in range(proposed.shape[0]):
            for second in range(first + 1, proposed.shape[0]):
                delta = proposed[first] - proposed[second]
                distance = torch.linalg.vector_norm(delta)
                penetration = torch.relu(min_separation - distance)
                direction = delta / torch.clamp(distance, min=1e-6)
                pair_correction = 0.5 * penetration * direction
                one_hot_first = F.one_hot(
                    torch.tensor(first, device=DEVICE),
                    num_classes=proposed.shape[0],
                ).to(proposed.dtype)
                one_hot_second = F.one_hot(
                    torch.tensor(second, device=DEVICE),
                    num_classes=proposed.shape[0],
                ).to(proposed.dtype)
                correction = correction + (
                    one_hot_first.unsqueeze(1)
                    - one_hot_second.unsqueeze(1)
                ) * pair_correction.unsqueeze(0)
        proposed = proposed + correction
    return torch.clamp(proposed, min=-5.0, max=5.0)


def rollout_objective(
    policy: FleetPolicy,
    scenario: Scenario,
    model: RateWorldModel,
    horizon: int,
    action_noise: ActionNoise,
    channel_noise: torch.Tensor,
    keep_trace: bool = False,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    positions = scenario.start_positions.clone()
    backlog = scenario.tasks.clone()
    initial_total = torch.sum(scenario.tasks)
    objective = torch.zeros((), device=DEVICE)
    trace: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for step in range(horizon):
        state_positions = positions
        state_backlog = backlog
        angle, speed, allocation = policy(
            positions,
            scenario.user_positions,
            backlog,
            scenario.obstacles,
            max_step=MAX_STEP,
        )
        angle = angle + action_noise.angle[step].reshape_as(angle)
        speed = torch.clamp(
            speed + action_noise.speed[step].reshape_as(speed),
            0.0,
            MAX_STEP,
        )
        logits = torch.log(allocation.reshape(-1) + 1e-8)
        allocation = F.softmax(
            logits + action_noise.allocation[step].reshape(-1), dim=0
        ).view_as(allocation)
        positions = safety_projected_advance(
            positions,
            angle,
            speed,
            scenario.obstacles,
        )
        rates = model.rates_at_step(
            positions,
            scenario.user_positions,
            step,
        )
        rates = rates * torch.exp(
            CHANNEL_NOISE_STD * channel_noise[step]
            - 0.5 * CHANNEL_NOISE_STD**2
        )
        delivered = (
            allocation * scenario.tasks.numel() * rates * TIME_STEP
        ).sum(dim=0)
        backlog = torch.clamp(backlog - delivered, min=0.0)
        objective = objective + torch.sum(backlog) / initial_total
        objective = objective + constraint_penalty(
            positions,
            scenario.obstacles,
            OBSTACLE_WEIGHT,
        )
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


def branch_model_matrix(
    policy: FleetPolicy,
    scenario: Scenario,
    branches: list[ActionNoise],
    channel_noises: list[torch.Tensor],
    horizon: int,
    models: list[RateWorldModel],
    keep_traces: bool,
) -> tuple[
    torch.Tensor,
    list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
]:
    rows: list[torch.Tensor] = []
    traces: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    central_index = len(models) // 2
    for branch, channel_noise in zip(branches, channel_noises):
        losses: list[torch.Tensor] = []
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
                channel_noise,
                keep_trace=keep_traces and model_index == central_index,
            )
            losses.append(loss)
            if trace:
                selected_trace = trace
        rows.append(torch.stack(losses))
        traces.append(selected_trace)
    return torch.stack(rows), traces


def robust_score(model_losses: torch.Tensor) -> torch.Tensor:
    score = torch.mean(model_losses)
    if model_losses.numel() > 1:
        score = score + ROBUSTNESS_WEIGHT * torch.std(
            model_losses, unbiased=False
        )
    return score


def policy_descriptor_parameters(policy: FleetPolicy) -> list[torch.Tensor]:
    layers = (policy.angle_head, policy.speed_head, policy.alloc_head)
    return [parameter for layer in layers for parameter in layer.parameters()]


def _flatten_gradients(
    gradients: Iterable[torch.Tensor | None],
    parameters: list[torch.Tensor],
) -> torch.Tensor:
    pieces = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(gradient.reshape(-1))
    return torch.cat(pieces).detach()


def branch_gradient_descriptors(
    policy: FleetPolicy,
    scenario: Scenario,
    branches: list[ActionNoise],
    channel_noises: list[torch.Tensor],
    horizon: int,
    models: list[RateWorldModel],
) -> torch.Tensor:
    parameters = policy_descriptor_parameters(policy)

    def one_gradient(
        branch: ActionNoise, channel_noise: torch.Tensor
    ) -> torch.Tensor:
        losses = []
        for model in models:
            loss, _ = rollout_objective(
                policy,
                scenario,
                model,
                horizon,
                branch,
                channel_noise,
            )
            losses.append(loss)
        score = robust_score(torch.stack(losses))
        gradients = torch.autograd.grad(
            score,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        return _flatten_gradients(gradients, parameters)

    baseline_gradient = one_gradient(branches[0], channel_noises[0])
    descriptors = [
        one_gradient(branch, noise) - baseline_gradient
        for branch, noise in zip(branches[1:], channel_noises[1:])
    ]
    return torch.stack(descriptors)


def branch_member_gradient_statistics(
    policy: FleetPolicy,
    scenario: Scenario,
    branches: list[ActionNoise],
    channel_noises: list[torch.Tensor],
    horizon: int,
    models: list[RateWorldModel],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ensemble directional agreement and the anchor loss gradient.

    Agreement is computed from per-model gradient increments before the
    ensemble is collapsed into one robust objective.  It is therefore a
    genuinely gradient-level reliability measure rather than a second copy of
    the objective-difference confidence.
    """
    parameters = policy_descriptor_parameters(policy)
    gradients_by_branch: list[torch.Tensor] = []
    for branch, channel_noise in zip(branches, channel_noises):
        member_gradients = []
        for model in models:
            loss, _ = rollout_objective(
                policy,
                scenario,
                model,
                horizon,
                branch,
                channel_noise,
            )
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            member_gradients.append(_flatten_gradients(gradients, parameters))
        gradients_by_branch.append(torch.stack(member_gradients))

    stacked = torch.stack(gradients_by_branch)
    anchor_members = stacked[0]
    increments = stacked[1:] - anchor_members.unsqueeze(0)
    mean_increment = torch.mean(increments, dim=1)
    numerator = torch.sum(mean_increment**2, dim=1)
    denominator = torch.mean(torch.sum(increments**2, dim=2), dim=1)
    agreement = numerator / torch.clamp(denominator, min=1e-8)
    agreement = torch.clamp(agreement, min=0.0, max=1.0)
    anchor_gradient = torch.mean(anchor_members, dim=0)
    return agreement.detach(), anchor_gradient.detach()


def soft_branch_label_probabilities(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth positive/negative probabilities for acquisition gradients."""
    deltas = matrix[1:] - matrix[0].unsqueeze(0)
    delta_mean = torch.mean(deltas, dim=1)
    delta_std = torch.std(deltas, dim=1, unbiased=False)
    threshold = max(0.005, 0.00025 * abs(float(torch.mean(matrix[0]))))
    temperature = 0.025 + delta_std
    logits = torch.stack(
        [
            (-delta_mean - threshold) / temperature,
            torch.zeros_like(delta_mean),
            (delta_mean - threshold) / temperature,
        ],
        dim=1,
    )
    probabilities = torch.softmax(logits, dim=1)
    return probabilities[:, 0].detach(), probabilities[:, 2].detach()


def branch_update_descriptors(
    policy: FleetPolicy,
    scenario: Scenario,
    traces: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Describe a branch by the gradient that its eventual local update uses."""
    parameters = policy_descriptor_parameters(policy)
    positive_probability, negative_probability = (
        soft_branch_label_probabilities(matrix)
    )
    descriptors = []
    for index, trace in enumerate(traces[1:]):
        positive_loss = local_distillation(
            policy,
            scenario,
            trace,
            positive=True,
        )
        negative_loss = local_distillation(
            policy,
            scenario,
            trace,
            positive=False,
        )
        expected_update_loss = (
            positive_probability[index] * POSITIVE_WEIGHT * positive_loss
            + negative_probability[index]
            * NEGATIVE_REJECTION_WEIGHT
            * negative_loss
        )
        gradients = torch.autograd.grad(
            expected_update_loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        descriptors.append(_flatten_gradients(gradients, parameters))
    return torch.stack(descriptors)


def reliable_useful_weights(
    descriptors: torch.Tensor,
    objective_confidence: torch.Tensor,
    gradient_agreement: torch.Tensor,
    anchor_gradient: torch.Tensor,
) -> torch.Tensor:
    """Combine reliability, magnitude, and descent-direction compatibility."""
    norms = torch.linalg.vector_norm(descriptors, dim=1)
    positive_norms = norms[norms > 1e-8]
    if positive_norms.numel() > 0:
        scale = torch.median(positive_norms)
    else:
        scale = torch.ones((), device=descriptors.device)
    magnitude = torch.tanh(norms / torch.clamp(scale, min=1e-8))
    anchor_norm = torch.linalg.vector_norm(anchor_gradient)
    cosine = torch.sum(
        descriptors * anchor_gradient.unsqueeze(0),
        dim=1,
    ) / torch.clamp(norms * anchor_norm, min=1e-8)
    compatibility = torch.clamp(cosine, min=0.0, max=1.0)
    compatibility = 0.10 + 0.90 * compatibility
    weights = (
        objective_confidence
        * gradient_agreement
        * magnitude
        * compatibility
    )
    return torch.clamp(weights, min=1e-4, max=1.0).detach()


def action_noise_descriptors(branches: list[ActionNoise]) -> torch.Tensor:
    descriptors = []
    for branch in branches[1:]:
        descriptors.append(
            torch.cat(
                [
                    (branch.angle / math.pi).reshape(-1),
                    (branch.speed / MAX_STEP).reshape(-1),
                    branch.allocation.reshape(-1),
                ]
            )
        )
    return torch.stack(descriptors)


def greedy_information_selection(
    descriptors: torch.Tensor,
    count: int,
    weights: torch.Tensor | None = None,
    initial_selected: list[int] | None = None,
    adaptive_min_gain: float | None = None,
    minimum_count: int = 1,
) -> tuple[list[int], float]:
    count = min(count, descriptors.shape[0])
    norms = torch.linalg.vector_norm(descriptors, dim=1, keepdim=True)
    normalized = descriptors / torch.clamp(norms, min=1e-8)
    if weights is not None:
        normalized = normalized * torch.sqrt(
            torch.clamp(weights, min=0.02, max=1.0)
        ).unsqueeze(-1)
    selected = list(initial_selected or [])
    if len(selected) > count:
        selected = selected[:count]
    remaining = [
        index for index in range(descriptors.shape[0]) if index not in selected
    ]
    final_logdet = 0.0
    if selected:
        initial_matrix = normalized[selected]
        initial_gram = initial_matrix @ initial_matrix.T
        initial_regularizer = (
            1.0 if adaptive_min_gain is not None else 1e-3
        )
        initial_sign, initial_value = torch.linalg.slogdet(
            initial_gram
            + initial_regularizer
            * torch.eye(len(selected), device=DEVICE)
        )
        if float(initial_sign) > 0:
            final_logdet = float(initial_value)
    while len(selected) < count:
        best_index = remaining[0]
        best_value = -float("inf")
        for candidate in remaining:
            proposal = selected + [candidate]
            matrix = normalized[proposal]
            gram = matrix @ matrix.T
            regularizer = (
                1.0 if adaptive_min_gain is not None else 1e-3
            )
            sign, logdet = torch.linalg.slogdet(
                gram
                + regularizer
                * torch.eye(len(proposal), device=DEVICE)
            )
            value = float(logdet) if float(sign) > 0 else -float("inf")
            if value > best_value:
                best_value = value
                best_index = candidate
        marginal_gain = best_value - final_logdet
        if (
            adaptive_min_gain is not None
            and len(selected) >= minimum_count
            and marginal_gain < adaptive_min_gain
        ):
            break
        selected.append(best_index)
        remaining.remove(best_index)
        final_logdet = best_value
    return selected, final_logdet


def information_logdet(
    descriptors: torch.Tensor,
    selected: list[int],
    weights: torch.Tensor | None = None,
) -> float:
    if not selected:
        return float("nan")
    matrix = descriptors[selected]
    matrix = matrix / torch.clamp(
        torch.linalg.vector_norm(matrix, dim=1, keepdim=True), min=1e-8
    )
    if weights is not None:
        matrix = matrix * torch.sqrt(
            torch.clamp(weights[selected], min=0.02, max=1.0)
        ).unsqueeze(-1)
    gram = matrix @ matrix.T
    # Report the same normalized information objective used by the adaptive
    # selector, F(S)=log det(I+G_S G_S^T).  The previous 1e-3 numerical
    # regularizer made valid diversity scores negative and obscured their
    # interpretation.
    sign, logdet = torch.linalg.slogdet(
        gram + torch.eye(len(selected), device=DEVICE)
    )
    return float(logdet) if float(sign) > 0 else float("nan")


def greedy_task_information_selection(
    descriptors: torch.Tensor,
    count: int,
    weights: torch.Tensor,
    pool_matrix: torch.Tensor,
    utility_weight: float,
    lcb_beta: float,
) -> list[int]:
    """Greedily maximize reliable gradient information plus task LCB.

    The log-determinant term rewards complementary update directions.  The
    nonnegative modular term rewards branches whose ensemble lower-confidence
    improvement is positive, preventing geometric diversity from displacing
    task relevance.
    """
    count = min(count, descriptors.shape[0])
    norms = torch.linalg.vector_norm(descriptors, dim=1, keepdim=True)
    normalized = descriptors / torch.clamp(norms, min=1e-8)
    normalized = normalized * torch.sqrt(
        torch.clamp(weights, min=0.02, max=1.0)
    ).unsqueeze(-1)
    improvements = pool_matrix[0].unsqueeze(0) - pool_matrix[1:]
    mean_improvement = torch.mean(improvements, dim=1)
    std_improvement = torch.std(improvements, dim=1, unbiased=False)
    lcb = mean_improvement - lcb_beta * std_improvement
    positive_scale = torch.median(torch.abs(mean_improvement))
    utility = torch.clamp(
        torch.relu(lcb) / torch.clamp(positive_scale, min=0.02),
        min=0.0,
        max=1.0,
    )

    selected: list[int] = []
    remaining = list(range(descriptors.shape[0]))
    while len(selected) < count:
        best_index = remaining[0]
        best_value = -float("inf")
        for candidate in remaining:
            proposal = selected + [candidate]
            matrix = normalized[proposal]
            gram = matrix @ matrix.T
            sign, logdet = torch.linalg.slogdet(
                gram + torch.eye(len(proposal), device=DEVICE)
            )
            information = (
                float(logdet) if float(sign) > 0 else -float("inf")
            )
            task_value = utility_weight * float(torch.sum(utility[proposal]))
            value = information + task_value
            if value > best_value:
                best_value = value
                best_index = candidate
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def greedy_residual_calibrated_selection(
    descriptors: torch.Tensor,
    count: int,
    weights: torch.Tensor,
    pool_matrix: torch.Tensor,
    utility_weight: float,
    lcb_beta: float,
    calibration_correction: float = 0.0,
) -> list[int]:
    """Residual-calibrated credible gradient information exploration.

    Extends TC-GIB with a data-driven correction for systematic WM bias.

    Objective:
        F(S) = log det(I + Σ_{b∈S} r_b ḡ_b ḡ_b^T)
               + λ Σ_{b∈S} [LCB_b]_+

    where the calibrated task LCB is:
        LCB_b = E_m[ΔJ_{b,m}] − β · Std_m[ΔJ_{b,m}] − q_{1−δ}

    and q_{1−δ} = −quantile(cache errors, δ) is the correction supplied
    by a CalibrationErrorCache.  When the entire ensemble over-predicts
    in the same direction the variance term β·σ is useless; q_{1−δ}
    catches that systematic component so only branches with credibly
    positive task value survive the relu gate.

    The same high-fidelity query that detects the bias simultaneously
    (a) corrects branch labels, (b) updates the WM ensemble, and
    (c) updates the calibration cache — three uses of one oracle call.
    """
    count = min(count, descriptors.shape[0])
    norms = torch.linalg.vector_norm(descriptors, dim=1, keepdim=True)
    normalized = descriptors / torch.clamp(norms, min=1e-8)
    normalized = normalized * torch.sqrt(
        torch.clamp(weights, min=0.02, max=1.0)
    ).unsqueeze(-1)

    improvements = pool_matrix[0].unsqueeze(0) - pool_matrix[1:]
    mean_improvement = torch.mean(improvements, dim=1)
    std_improvement = torch.std(improvements, dim=1, unbiased=False)

    # Calibrated LCB: subtract the residual quantile correction so that
    # systematic WM over-prediction lowers the bound along with variance.
    lcb = (
        mean_improvement
        - lcb_beta * std_improvement
        - calibration_correction
    )

    # Normalise to [0, 1] using the median absolute improvement scale so
    # the task bonus is commensurable with the log-det information gain.
    positive_scale = torch.median(torch.abs(mean_improvement))
    utility = torch.clamp(
        torch.relu(lcb) / torch.clamp(positive_scale, min=0.02),
        min=0.0,
        max=1.0,
    )

    selected: list[int] = []
    remaining = list(range(descriptors.shape[0]))
    while len(selected) < count:
        best_index = remaining[0]
        best_value = -float("inf")
        for candidate in remaining:
            proposal = selected + [candidate]
            matrix = normalized[proposal]
            gram = matrix @ matrix.T
            sign, logdet = torch.linalg.slogdet(
                gram + torch.eye(len(proposal), device=DEVICE)
            )
            information = (
                float(logdet) if float(sign) > 0 else -float("inf")
            )
            task_value = utility_weight * float(torch.sum(utility[proposal]))
            value = information + task_value
            if value > best_value:
                best_value = value
                best_index = candidate
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def select_branch_indices(
    variant: Variant,
    gradient_descriptors: torch.Tensor,
    action_descriptors: torch.Tensor,
    pool_matrix: torch.Tensor,
    selection_confidence: torch.Tensor,
    count: int,
    seed: int,
    calibration_error_cache: "CalibrationErrorCache | None" = None,
) -> list[int]:
    """Select equal-budget branches with an explicitly named criterion."""
    strategy = variant.selection_strategy
    if not strategy:
        strategy = (
            "gradient_information"
            if variant.gradient_information
            else "action_entropy"
        )
    if strategy == "random":
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed)
        return (
            torch.randperm(
                gradient_descriptors.shape[0],
                generator=generator,
                device=DEVICE,
            )[:count]
            .detach()
            .cpu()
            .tolist()
        )
    if strategy == "action_entropy":
        selected, _ = greedy_information_selection(
            action_descriptors,
            count,
        )
        return selected
    if strategy == "ensemble_ucb":
        improvements = (
            pool_matrix[0].unsqueeze(0) - pool_matrix[1:]
        )
        score = torch.mean(improvements, dim=1) + variant.ucb_beta * torch.std(
            improvements,
            dim=1,
            unbiased=False,
        )
        return (
            torch.topk(score, k=min(count, score.numel())).indices
            .detach()
            .cpu()
            .tolist()
        )
    if strategy == "residual_calibrated_gib":
        correction = (
            calibration_error_cache.correction(variant.calib_delta)
            if calibration_error_cache is not None
            else 0.0
        )
        return greedy_residual_calibrated_selection(
            gradient_descriptors,
            count,
            selection_confidence,
            pool_matrix,
            variant.task_utility_weight,
            variant.lcb_beta,
            calibration_correction=correction,
        )
    if strategy == "task_gradient_information":
        return greedy_task_information_selection(
            gradient_descriptors,
            count,
            selection_confidence,
            pool_matrix,
            variant.task_utility_weight,
            variant.lcb_beta,
        )
    if strategy == "dual_anchor_gradient_information":
        action_anchor, _ = greedy_information_selection(
            action_descriptors,
            1,
        )
        improvements = pool_matrix[0].unsqueeze(0) - pool_matrix[1:]
        lower_confidence = torch.mean(improvements, dim=1) - (
            variant.lcb_beta
            * torch.std(improvements, dim=1, unbiased=False)
        )
        ranked_task = torch.argsort(
            lower_confidence, descending=True
        ).detach().cpu().tolist()
        anchors = list(action_anchor)
        for task_anchor in ranked_task:
            if task_anchor not in anchors:
                anchors.append(int(task_anchor))
                break
        anchors = anchors[: min(count, len(anchors))]
        selected, _ = greedy_information_selection(
            gradient_descriptors,
            count,
            weights=selection_confidence,
            initial_selected=anchors,
        )
        return selected
    if strategy == "gradient_information":
        selected, _ = greedy_information_selection(
            gradient_descriptors,
            count,
            weights=selection_confidence,
            adaptive_min_gain=(
                variant.adaptive_min_gain
                if variant.adaptive_selection
                else None
            ),
        )
        return selected
    if strategy == "gradient_information_anchored":
        candidate_scores = torch.stack(
            [robust_score(row) for row in pool_matrix[1:]]
        )
        anchor = int(torch.argmin(candidate_scores))
        selected, _ = greedy_information_selection(
            gradient_descriptors,
            count,
            weights=selection_confidence,
            initial_selected=[anchor],
            adaptive_min_gain=(
                variant.adaptive_min_gain
                if variant.adaptive_selection
                else None
            ),
        )
        return selected
    if strategy == "gradient_information_action_anchor":
        action_selected, _ = greedy_information_selection(
            action_descriptors,
            count,
        )
        selected, _ = greedy_information_selection(
            gradient_descriptors,
            count,
            weights=selection_confidence,
            initial_selected=action_selected[:1],
            adaptive_min_gain=(
                variant.adaptive_min_gain
                if variant.adaptive_selection
                else None
            ),
        )
        return selected
    raise ValueError(f"Unknown branch-selection strategy: {strategy}")


def selected_information_weights(
    descriptors: torch.Tensor,
    selected: list[int],
) -> torch.Tensor:
    """Measure each branch's gradient direction outside the selected span."""
    if not selected:
        return torch.empty(0, device=DEVICE)
    normalized = descriptors / torch.clamp(
        torch.linalg.vector_norm(descriptors, dim=1, keepdim=True),
        min=1e-8,
    )
    basis: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for index in selected:
        direction = normalized[index]
        residual = direction
        for vector in basis:
            residual = residual - torch.dot(residual, vector) * vector
        novelty = torch.clamp(torch.sum(residual**2), min=0.05, max=1.0)
        weights.append(novelty)
        residual_norm = torch.linalg.vector_norm(residual)
        if float(residual_norm) > 1e-6:
            basis.append(residual / residual_norm)
    return torch.stack(weights).detach()


def compute_trust_profile(
    trace: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    scenario: Scenario,
    models: list[RateWorldModel],
    horizon: int,
) -> torch.Tensor:
    values = []
    initial_total = torch.sum(scenario.tasks)
    with torch.no_grad():
        for step, (positions, backlog, _) in enumerate(trace):
            predictions = torch.stack(
                [
                    model.rates_at_step(
                        positions,
                        scenario.user_positions,
                        step,
                    )
                    for model in models
                ]
            )
            coefficient = torch.mean(
                torch.std(predictions, dim=0, unbiased=False)
                / (torch.mean(predictions, dim=0) + 1e-4)
            )
            load = torch.clamp(torch.sum(backlog) / initial_total, 0.0, 1.0)
            radius = (0.35 + 0.65 * torch.sqrt(load)) / (
                1.0 + 2.5 * coefficient
            )
            values.append(torch.clamp(radius, 0.20, 1.0))
    if not values:
        return torch.ones(horizon, device=DEVICE)
    profile = torch.stack(values)
    if profile.numel() < horizon:
        profile = torch.cat(
            [
                profile,
                profile[-1].repeat(horizon - profile.numel()),
            ]
        )
    return profile[:horizon]


def scale_branches(
    branches: list[ActionNoise], profile: torch.Tensor
) -> list[ActionNoise]:
    scaled = []
    for branch in branches:
        angle_profile = profile.reshape(
            profile.shape[0], *([1] * (branch.angle.ndim - 1))
        )
        speed_profile = profile.reshape(
            profile.shape[0], *([1] * (branch.speed.ndim - 1))
        )
        allocation_profile = profile.reshape(
            profile.shape[0], *([1] * (branch.allocation.ndim - 1))
        )
        scaled.append(
            ActionNoise(
                branch.angle * angle_profile,
                branch.speed * speed_profile,
                branch.allocation * allocation_profile,
            )
        )
    return scaled


def global_positive_distillation(
    policy: FleetPolicy,
    scenario: Scenario,
    trace: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if not trace:
        return torch.zeros((), device=DEVICE)
    losses = []
    stride = max(1, len(trace) // 20)
    for positions, backlog, target in trace[::stride]:
        angle, speed, allocation = policy(
            positions,
            scenario.user_positions,
            backlog,
            scenario.obstacles,
            max_step=MAX_STEP,
        )
        losses.append(
            torch.mean(
                (action_features(angle, speed, allocation) - target) ** 2
            )
        )
    return torch.mean(torch.stack(losses))


def local_distillation(
    policy: FleetPolicy,
    scenario: Scenario,
    trace: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    positive: bool,
) -> torch.Tensor:
    if not trace:
        return torch.zeros((), device=DEVICE)
    losses = []
    local_steps = min(len(trace), max(4, len(trace) // 3))
    for index, (positions, backlog, target) in enumerate(trace[:local_steps]):
        angle, speed, allocation = policy(
            positions,
            scenario.user_positions,
            backlog,
            scenario.obstacles,
            max_step=MAX_STEP,
        )
        distance = torch.mean(
            (action_features(angle, speed, allocation) - target) ** 2
        )
        temporal_weight = math.exp(-1.5 * index / max(local_steps - 1, 1))
        if positive:
            losses.append(temporal_weight * distance)
        else:
            losses.append(
                temporal_weight
                * F.relu(LOCAL_REJECTION_MARGIN - distance)
            )
    return torch.mean(torch.stack(losses))


def robust_labels(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    baseline = matrix[0]
    deltas = matrix[1:] - baseline.unsqueeze(0)
    threshold = max(0.005, 0.00025 * abs(float(torch.mean(baseline))))
    delta_mean = torch.mean(deltas, dim=1)
    delta_std = torch.std(deltas, dim=1, unbiased=False)
    upper = delta_mean + 0.50 * delta_std
    lower = delta_mean - 0.50 * delta_std
    labels = torch.zeros(deltas.shape[0], dtype=torch.int8, device=DEVICE)
    labels[upper < -threshold] = 1
    labels[lower > threshold] = -1
    scale = 0.10 + 0.02 * abs(float(torch.mean(baseline)))
    confidence = torch.zeros_like(upper)
    positive = labels == 1
    negative = labels == -1
    confidence[positive] = torch.clamp(
        (-upper[positive] - threshold) / scale, 0.0, 1.0
    )
    confidence[negative] = torch.clamp(
        (lower[negative] - threshold) / scale, 0.0, 1.0
    )
    return labels, confidence, threshold


def branch_confidence_weights(matrix: torch.Tensor) -> torch.Tensor:
    """Reliability weights for signed, model-disagreement-aware information."""
    deltas = matrix[1:] - matrix[0].unsqueeze(0)
    delta_mean = torch.mean(deltas, dim=1)
    delta_std = torch.std(deltas, dim=1, unbiased=False)
    sign_signal = torch.abs(delta_mean) / (delta_std + 0.02)
    magnitude = torch.tanh(torch.abs(delta_mean) / 0.05)
    reliability = magnitude * torch.tanh(sign_signal)
    return torch.clamp(reliability, min=0.02, max=1.0)


def true_labels(
    matrix: torch.Tensor, threshold: float
) -> torch.Tensor:
    deltas = matrix[1:, 0] - matrix[0, 0]
    labels = torch.zeros(deltas.shape[0], dtype=torch.int8, device=DEVICE)
    labels[deltas < -threshold] = 1
    labels[deltas > threshold] = -1
    return labels


def collect_high_fidelity_observations(
    policy: FleetPolicy,
    scenario: Scenario,
    branch: ActionNoise,
    horizon: int,
    buffer: CalibrationBuffer,
) -> None:
    positions = scenario.start_positions.clone()
    backlog = scenario.tasks.clone()
    with torch.no_grad():
        for step in range(min(horizon, MAX_QUERY_STEPS)):
            angle, speed, allocation = policy(
                positions,
                scenario.user_positions,
                backlog,
                scenario.obstacles,
                max_step=MAX_STEP,
            )
            angle = angle + branch.angle[step].reshape_as(angle)
            speed = torch.clamp(
                speed + branch.speed[step].reshape_as(speed),
                0.0,
                MAX_STEP,
            )
            logits = torch.log(allocation.reshape(-1) + 1e-8)
            allocation = F.softmax(
                logits + branch.allocation[step].reshape(-1), dim=0
            ).view_as(allocation)
            positions = safety_projected_advance(
                positions,
                angle,
                speed,
                scenario.obstacles,
            )
            rates = TRUE_MODEL.rates_at_step(
                positions,
                scenario.user_positions,
                step,
            )
            aav_pairs, user_pairs = _pairwise_geometry(
                positions, scenario.user_positions
            )
            buffer.add(aav_pairs, user_pairs, rates.reshape(-1))
            delivered = (
                allocation * scenario.tasks.numel() * rates * TIME_STEP
            ).sum(dim=0)
            backlog = torch.clamp(backlog - delivered, min=0.0)
            if float(torch.sum(backlog)) < 1e-3 * backlog.numel():
                break


def fit_calibrated_ensemble(
    models: list[RateWorldModel],
    buffer: CalibrationBuffer,
    seed: int,
) -> None:
    if len(buffer) < 8:
        return
    aav_xy, user_xy, targets = buffer.tensors()
    if targets.shape[0] > 1200:
        aav_xy = aav_xy[-1200:]
        user_xy = user_xy[-1200:]
        targets = targets[-1200:]
    for model_index, model in enumerate(models):
        parameters = list(model.parameters())
        if not parameters:
            continue
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed + 7919 * model_index)
        optimizer = torch.optim.Adam(
            parameters,
            lr=0.006 if isinstance(model, ResidualWorldModel) else 0.035,
        )
        steps = 42 if isinstance(model, ResidualWorldModel) else 12
        model.train()
        for _ in range(steps):
            sample_count = min(192, targets.shape[0])
            indices = torch.randint(
                targets.shape[0],
                (sample_count,),
                generator=generator,
                device=DEVICE,
            )
            prediction = model.pair_rates(
                aav_xy[indices], user_xy[indices]
            )
            loss = F.mse_loss(prediction, targets[indices])
            if isinstance(model, ResidualWorldModel):
                anchor_count = min(64, targets.shape[0])
                anchor_x = -4.5 + 9.0 * torch.rand(
                    anchor_count, generator=generator, device=DEVICE
                )
                anchor_y = torch.clamp(
                    1.5
                    * torch.randn(
                        anchor_count, generator=generator, device=DEVICE
                    ),
                    min=-3.5,
                    max=3.5,
                )
                anchor_aav = torch.stack([anchor_x, anchor_y], dim=1)
                anchor_indices = torch.randint(
                    user_xy.shape[0],
                    (anchor_count,),
                    generator=generator,
                    device=DEVICE,
                )
                anchor_user = user_xy[anchor_indices]
                anchor_target = _parametric_pair_rates(
                    anchor_aav,
                    anchor_user,
                    NOMINAL_SPEC.pathloss_exp,
                    NOMINAL_SPEC.shadow_strength,
                    NOMINAL_SPEC.shadow_width,
                )
                anchor_prediction = model.pair_rates(
                    anchor_aav, anchor_user
                )
                loss = loss + 0.03 * F.mse_loss(
                    anchor_prediction, anchor_target
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
        model.eval()


def make_holdout_pairs(
    scenario: Scenario,
    seed: int,
    count: int = 160,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    x_coord = -4.2 + 8.4 * torch.rand(
        count, generator=generator, device=DEVICE
    )
    y_coord = 1.15 * torch.randn(
        count, generator=generator, device=DEVICE
    )
    y_coord = torch.clamp(y_coord, min=-2.8, max=2.8)
    aav_xy = torch.stack([x_coord, y_coord], dim=1)
    indices = torch.randint(
        scenario.user_positions.shape[0],
        (count,),
        generator=generator,
        device=DEVICE,
    )
    user_xy = scenario.user_positions[indices]
    with torch.no_grad():
        target = TRUE_MODEL.pair_rates(aav_xy, user_xy)
    return aav_xy, user_xy, target


def ensemble_rmse(
    models: list[RateWorldModel],
    holdout: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    aav_xy, user_xy, target = holdout
    with torch.no_grad():
        prediction = torch.mean(
            torch.stack(
                [model.pair_rates(aav_xy, user_xy) for model in models]
            ),
            dim=0,
        )
        return float(torch.sqrt(F.mse_loss(prediction, target)))


def deterministic_evaluation(
    policy: FleetPolicy,
    scenario: Scenario,
    horizon: int,
) -> dict[str, float | np.ndarray]:
    positions = scenario.start_positions.clone()
    backlog = scenario.tasks.clone()
    initial_total = float(torch.sum(backlog))
    backlog_area = 0.0
    trajectory = [positions.detach().cpu().numpy().copy()]
    completed = False
    collision = False
    min_obstacle_clearance = float("inf")
    min_aav_separation = float("inf")
    steps = horizon
    with torch.no_grad():
        for step in range(horizon):
            angle, speed, allocation = policy(
                positions,
                scenario.user_positions,
                backlog,
                scenario.obstacles,
                max_step=MAX_STEP,
            )
            positions = safety_projected_advance(
                positions,
                angle,
                speed,
                scenario.obstacles,
            )
            if scenario.obstacles.numel():
                obstacle_distances = torch.linalg.vector_norm(
                    positions.unsqueeze(1)
                    - scenario.obstacles[:, :2].unsqueeze(0),
                    dim=-1,
                )
                obstacle_clearance = (
                    obstacle_distances
                    - scenario.obstacles[:, 2].unsqueeze(0)
                )
                min_obstacle_clearance = min(
                    min_obstacle_clearance,
                    float(torch.min(obstacle_clearance)),
                )
                collision = collision or bool(
                    torch.any(obstacle_clearance < 0.0)
                )
            if positions.shape[0] > 1:
                pairwise = torch.cdist(positions, positions)
                mask = torch.triu(
                    torch.ones_like(pairwise, dtype=torch.bool),
                    diagonal=1,
                )
                separation = pairwise[mask]
                if separation.numel():
                    min_aav_separation = min(
                        min_aav_separation,
                        float(torch.min(separation)),
                    )
                    collision = collision or bool(
                        torch.any(separation < 0.45)
                    )
            rates = TRUE_MODEL.rates_at_step(
                positions,
                scenario.user_positions,
                step,
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
        "completed": float(completed),
        "safe_completed": float(completed and not collision),
        "collision": float(collision),
        "min_obstacle_clearance": min_obstacle_clearance,
        "min_aav_separation": min_aav_separation,
        "backlog_area": backlog_area,
        "remaining": float(torch.sum(backlog)),
        "trajectory": np.concatenate(trajectory, axis=0),
    }


def policy_vector(policy: FleetPolicy) -> np.ndarray:
    return torch.cat(
        [parameter.detach().flatten().cpu() for parameter in policy.parameters()]
    ).numpy()


def train_variant(
    variant: Variant,
    scenario: Scenario,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, np.ndarray | float | FleetPolicy]:
    torch.manual_seed(seed)
    num_aavs = int(scenario.start_positions.shape[0])
    num_obstacles = int(scenario.obstacles.shape[0])
    policy = FleetPolicy(
        scenario.tasks.numel(),
        num_aavs=num_aavs,
        num_obstacles=num_obstacles,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=1e-3, weight_decay=0.0
    )
    models = build_ensemble(variant, seed + 300_000)
    calibration_buffer = CalibrationBuffer.empty()
    calibration_error_cache: CalibrationErrorCache | None = (
        CalibrationErrorCache(max_samples=variant.calib_max_samples)
        if variant.use_residual_calibration
        else None
    )
    holdout = make_holdout_pairs(scenario, seed + 400_000)

    training_objective = np.zeros(config.train_episodes)
    nominal_objective = np.zeros(config.train_episodes)
    update_norm = np.zeros(config.train_episodes)
    gradient_logdet = np.zeros(config.train_episodes)
    action_logdet = np.zeros(config.train_episodes)
    positive_counts = np.zeros(config.train_episodes)
    negative_counts = np.zeros(config.train_episodes)
    ambiguous_counts = np.zeros(config.train_episodes)
    query_counts = np.zeros(config.train_episodes)
    high_fidelity_observation_counts = np.zeros(config.train_episodes)
    wm_rollout_calls = np.zeros(config.train_episodes)
    selected_counts = np.zeros(config.train_episodes)
    candidate_counts = np.zeros(config.train_episodes)
    selected_reliability = np.zeros(config.train_episodes)
    effective_trial_weights = np.zeros(config.train_episodes)
    wm_line_search_gain = np.zeros(config.train_episodes)
    model_rmse = np.zeros(config.train_episodes + 1)
    predicted_labels = np.zeros(
        (config.train_episodes, config.selected_branches), dtype=np.int8
    )
    audited_labels = np.full_like(predicted_labels, 2)
    selected_indices = np.full(
        (config.train_episodes, config.selected_branches), dtype=np.int16
        , fill_value=-1
    )
    information_weights = np.zeros(
        (config.train_episodes, config.selected_branches), dtype=np.float32
    )
    model_rmse[0] = ensemble_rmse(models, holdout)

    checkpoint_episodes: list[int] = []
    checkpoint_steps: list[float] = []
    checkpoint_backlog: list[float] = []
    start_time = time.perf_counter()
    best_wm_score = float("inf")
    best_checkpoint_episode = 0
    best_policy_state: dict[str, torch.Tensor] | None = None
    best_hf_score = float("inf")
    best_hf_checkpoint_episode = 0
    best_hf_policy_state: dict[str, torch.Tensor] | None = None
    validation_branch = make_action_branches(
        seed + 950_000,
        1,
        config.train_horizon,
        scenario.tasks.numel(),
        num_aavs=num_aavs,
    )[0]
    validation_noises = [
        make_channel_noise(
            seed + 960_000 + 104_729 * repeat,
            config.train_horizon,
            num_aavs,
            scenario.tasks.numel(),
        )
        for repeat in range(max(1, variant.line_search_noise_repeats))
    ]
    wm_validation_call_count = len(models) * len(validation_noises)

    def evaluate_wm_score() -> float:
        with torch.no_grad():
            noise_scores = []
            for validation_noise in validation_noises:
                validation_objectives = []
                for validation_model in models:
                    validation_objective, _ = rollout_objective(
                        policy,
                        scenario,
                        validation_model,
                        config.train_horizon,
                        validation_branch,
                        validation_noise,
                    )
                    validation_objectives.append(validation_objective)
                noise_scores.append(
                    robust_score(torch.stack(validation_objectives))
                )
            stacked_scores = torch.stack(noise_scores)
            score = torch.mean(stacked_scores)
            if stacked_scores.numel() > 1:
                score = score + ROBUSTNESS_WEIGHT * torch.std(
                    stacked_scores, unbiased=False
                )
        return float(score)

    def update_wm_checkpoint(episode_index: int) -> int:
        """Keep the best policy under a fixed, ensemble-robust WM audit."""
        nonlocal best_wm_score, best_checkpoint_episode, best_policy_state
        if not variant.wm_checkpoint_selection:
            return 0
        score = evaluate_wm_score()
        if score < best_wm_score:
            best_wm_score = score
            best_checkpoint_episode = episode_index + 1
            best_policy_state = copy.deepcopy(policy.state_dict())
        return wm_validation_call_count

    for episode in range(config.train_episodes):
        episode_wm_calls = 0.0
        branch_seed = seed * 100_000 + episode
        branches = make_action_branches(
            branch_seed,
            config.pool_branches + 1,
            config.train_horizon,
            scenario.tasks.numel(),
            num_aavs=num_aavs,
        )
        active_pool_branches = config.pool_branches
        if variant.adaptive_pool:
            if episode == 0:
                active_pool_branches = variant.minimum_pool_branches
            else:
                active_pool_branches = max(
                    variant.minimum_pool_branches,
                    int(selected_counts[episode - 1]) + 2,
                )
            active_pool_branches = min(
                config.pool_branches,
                active_pool_branches,
            )
        branches = branches[: active_pool_branches + 1]
        candidate_counts[episode] = active_pool_branches
        if variant.common_random_numbers:
            noise_seed = (
                seed + 970_000
                if variant.persistent_crn
                else branch_seed + 50_000
            )
            shared_noise = make_channel_noise(
                noise_seed,
                config.train_horizon,
                num_aavs,
                scenario.tasks.numel(),
            )
            channel_noises = [shared_noise for _ in branches]
        else:
            channel_noises = [
                make_channel_noise(
                    branch_seed + 50_000 + 101 * index,
                    config.train_horizon,
                    num_aavs,
                    scenario.tasks.numel(),
                )
                for index in range(len(branches))
            ]

        if variant.selection_strategy in (
            "single_trial",
            "single_trial_robust",
        ):
            # A genuine single-trial baseline: optimize only the nominal
            # trajectory.  Candidate branches, ensemble labels, and audit
            # rollouts are intentionally skipped rather than merely hidden
            # from the accounting.
            if variant.selection_strategy == "single_trial_robust":
                member_objectives = []
                for model in models:
                    member_objective, _ = rollout_objective(
                        policy,
                        scenario,
                        model,
                        config.train_horizon,
                        branches[0],
                        channel_noises[0],
                    )
                    member_objectives.append(member_objective)
                baseline_objective = robust_score(
                    torch.stack(member_objectives)
                )
                episode_wm_calls = float(len(models))
            else:
                baseline_objective, _ = rollout_objective(
                    policy,
                    scenario,
                    NOMINAL_MODEL,
                    config.train_horizon,
                    branches[0],
                    channel_noises[0],
                )
                episode_wm_calls = 1.0
            optimizer.zero_grad()
            baseline_objective.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=1.0
            )
            optimizer.step()
            episode_wm_calls += update_wm_checkpoint(episode)
            objective_value = float(baseline_objective.detach())
            if not np.isfinite(objective_value):
                raise FloatingPointError(
                    f"Non-finite objective for {variant.name}, "
                    f"episode {episode}."
                )
            training_objective[episode] = objective_value
            nominal_objective[episode] = objective_value
            update_norm[episode] = float(grad_norm)
            model_rmse[episode + 1] = ensemble_rmse(models, holdout)
            wm_rollout_calls[episode] = episode_wm_calls

            if (
                episode % config.checkpoint_every == 0
                or episode == config.train_episodes - 1
            ):
                evaluation = deterministic_evaluation(
                    policy, scenario, config.eval_horizon
                )
                checkpoint_episodes.append(episode + 1)
                checkpoint_steps.append(float(evaluation["steps"]))
                checkpoint_backlog.append(float(evaluation["backlog_area"]))
            continue

        if variant.trust_region:
            with torch.no_grad():
                _, baseline_trace = rollout_objective(
                    policy,
                    scenario,
                    models[len(models) // 2],
                    config.train_horizon,
                    branches[0],
                    channel_noises[0],
                    keep_trace=True,
                )
            episode_wm_calls += 1.0
            trust_profile = compute_trust_profile(
                baseline_trace, scenario, models, config.train_horizon
            )
            branches = scale_branches(branches, trust_profile)

        # Every selector may reuse the already evaluated pool as its teacher.
        # Keeping traces for all methods prevents an implementation-level cache
        # difference from being mistaken for an algorithmic efficiency gain.
        keep_pool_traces = True
        with torch.no_grad():
            pool_matrix, pool_traces = branch_model_matrix(
                policy,
                scenario,
                branches,
                channel_noises,
                config.train_horizon,
                models,
                keep_traces=keep_pool_traces,
            )
        episode_wm_calls += len(branches) * len(models)

        if variant.update_aligned_descriptor:
            gradient_descriptors = branch_update_descriptors(
                policy,
                scenario,
                pool_traces,
                pool_matrix,
            )
        else:
            gradient_descriptors = branch_gradient_descriptors(
                policy,
                scenario,
                branches,
                channel_noises,
                config.train_horizon,
                models,
            )
            episode_wm_calls += len(branches) * len(models)

        if variant.multi_horizon_descriptor:
            short_horizon = max(4, config.train_horizon // 2)
            with torch.no_grad():
                short_matrix, short_traces = branch_model_matrix(
                    policy,
                    scenario,
                    branches,
                    channel_noises,
                    short_horizon,
                    models,
                    keep_traces=True,
                )
            episode_wm_calls += len(branches) * len(models)
            short_descriptors = branch_update_descriptors(
                policy,
                scenario,
                short_traces,
                short_matrix,
            )
            full_normalized = gradient_descriptors / torch.clamp(
                torch.linalg.vector_norm(
                    gradient_descriptors, dim=1, keepdim=True
                ),
                min=1e-8,
            )
            short_normalized = short_descriptors / torch.clamp(
                torch.linalg.vector_norm(
                    short_descriptors, dim=1, keepdim=True
                ),
                min=1e-8,
            )
            gradient_descriptors = torch.cat(
                [short_normalized, full_normalized],
                dim=1,
            )

        action_descriptors = action_noise_descriptors(branches)
        selection_confidence = branch_confidence_weights(pool_matrix)
        if variant.reliable_useful_weighting:
            gradient_agreement, anchor_gradient = (
                branch_member_gradient_statistics(
                    policy,
                    scenario,
                    branches,
                    channel_noises,
                    config.train_horizon,
                    models,
                )
            )
            episode_wm_calls += len(branches) * len(models)
            if variant.multi_horizon_descriptor:
                anchor_gradient = torch.cat(
                    [anchor_gradient, anchor_gradient],
                    dim=0,
                )
            selection_confidence = reliable_useful_weights(
                gradient_descriptors,
                selection_confidence,
                gradient_agreement,
                anchor_gradient,
            )
        selected = select_branch_indices(
            variant,
            gradient_descriptors,
            action_descriptors,
            pool_matrix,
            selection_confidence,
            config.selected_branches,
            branch_seed + 60_000,
            calibration_error_cache=calibration_error_cache,
        )
        selected_count = len(selected)
        selected_counts[episode] = selected_count
        if selected_count:
            selected_reliability[episode] = float(
                torch.mean(selection_confidence[selected])
            )
        selected_indices[episode, :selected_count] = np.asarray(
            selected, dtype=np.int16
        )
        selected_novelty = selected_information_weights(
            gradient_descriptors,
            selected,
        )
        information_weights[episode, :selected_count] = (
            selected_novelty.detach().cpu().numpy()
        )
        gradient_logdet[episode] = information_logdet(
            gradient_descriptors,
            selected,
            weights=selection_confidence,
        )
        action_logdet[episode] = information_logdet(
            action_descriptors, selected
        )

        selected_branches = [branches[0]] + [
            branches[index + 1] for index in selected
        ]
        selected_noises = [channel_noises[0]] + [
            channel_noises[index + 1] for index in selected
        ]

        pool_rows = [0] + [index + 1 for index in selected]
        teacher_matrix = pool_matrix[pool_rows]
        traces = [pool_traces[index] for index in pool_rows]
        labels, confidence, threshold = robust_labels(teacher_matrix)

        if variant.high_fidelity_query:
            ambiguous = torch.nonzero(labels == 0, as_tuple=False).reshape(-1)
            query_candidates = ambiguous
            if query_candidates.numel() == 0 and variant.force_query:
                query_candidates = torch.arange(
                    labels.numel(), device=DEVICE
                )
            if (
                query_candidates.numel() == 0
                and variant.learned_residual
                and episode < 4
            ):
                # A learned residual needs a small calibration warm-up before
                # its own confidence can decide whether a branch is ambiguous.
                query_candidates = torch.arange(
                    labels.numel(), device=DEVICE
                )
            if query_candidates.numel() > 0:
                deltas = (
                    teacher_matrix[1:] - teacher_matrix[0].unsqueeze(0)
                )
                disagreement = torch.std(
                    deltas, dim=1, unbiased=False
                )
                query_score = disagreement
                if variant.task_aware_query:
                    descriptor_norm = torch.linalg.vector_norm(
                        gradient_descriptors[selected],
                        dim=1,
                    )
                    descriptor_norm = descriptor_norm / torch.clamp(
                        torch.max(descriptor_norm),
                        min=1e-8,
                    )
                    query_score = (
                        disagreement
                        * (0.25 + 0.75 * selected_novelty)
                        * (0.25 + 0.75 * descriptor_norm)
                    )
                query_local = int(
                    query_candidates[
                        torch.argmax(query_score[query_candidates])
                    ]
                )
                buffer_before = len(calibration_buffer)
                collect_high_fidelity_observations(
                    policy,
                    scenario,
                    selected_branches[0],
                    config.train_horizon,
                    calibration_buffer,
                )
                collect_high_fidelity_observations(
                    policy,
                    scenario,
                    selected_branches[query_local + 1],
                    config.train_horizon,
                    calibration_buffer,
                )
                fit_calibrated_ensemble(
                    models,
                    calibration_buffer,
                    branch_seed + 70_000,
                )
                query_counts[episode] = 1.0
                high_fidelity_observation_counts[episode] = (
                    len(calibration_buffer) - buffer_before
                )
                with torch.no_grad():
                    teacher_matrix, traces = branch_model_matrix(
                        policy,
                        scenario,
                        selected_branches,
                        selected_noises,
                        config.train_horizon,
                        models,
                        keep_traces=True,
                    )
                episode_wm_calls += len(selected_branches) * len(models)
                labels, confidence, threshold = robust_labels(
                    teacher_matrix
                )
                with torch.no_grad():
                    queried_true_matrix, _ = branch_model_matrix(
                        policy,
                        scenario,
                        [
                            selected_branches[0],
                            selected_branches[query_local + 1],
                        ],
                        [
                            selected_noises[0],
                            selected_noises[query_local + 1],
                        ],
                        config.train_horizon,
                        [TRUE_MODEL],
                        keep_traces=False,
                    )
                    queried_delta = (
                        queried_true_matrix[1, 0]
                        - queried_true_matrix[0, 0]
                    )
                    if calibration_error_cache is not None:
                        # Record signed prediction error for the queried branch.
                        # improvement = baseline_loss − branch_loss (positive = better).
                        # Post-fit teacher_matrix already reflects the updated ensemble.
                        wm_improvement = float(
                            torch.mean(
                                teacher_matrix[0]
                                - teacher_matrix[query_local + 1]
                            )
                        )
                        true_improvement = -float(queried_delta)
                        calibration_error_cache.record(
                            true_improvement, wm_improvement
                        )
                    if variant.hf_checkpoint_selection:
                        queried_baseline_score = float(
                            queried_true_matrix[0, 0]
                        )
                        if queried_baseline_score < best_hf_score:
                            best_hf_score = queried_baseline_score
                            best_hf_checkpoint_episode = episode
                            best_hf_policy_state = copy.deepcopy(
                                policy.state_dict()
                            )
                    if float(queried_delta) < -threshold:
                        labels[query_local] = 1
                    elif float(queried_delta) > threshold:
                        labels[query_local] = -1
                    else:
                        labels[query_local] = 0
                    confidence[query_local] = 1.0

        positive_counts[episode] = float(torch.sum(labels == 1))
        negative_counts[episode] = float(torch.sum(labels == -1))
        ambiguous_counts[episode] = float(torch.sum(labels == 0))
        predicted_labels[episode, :selected_count] = (
            labels.detach().cpu().numpy()
        )

        baseline_objective, _ = rollout_objective(
            policy,
            scenario,
            NOMINAL_MODEL,
            config.train_horizon,
            selected_branches[0],
            selected_noises[0],
        )
        episode_wm_calls += 1.0
        objective = baseline_objective
        nominal_objective[episode] = float(baseline_objective.detach())

        if variant.local_pn:
            positive_losses = []
            negative_losses = []
            for local_index, label in enumerate(labels.tolist()):
                trace = traces[local_index + 1]
                weight = confidence[local_index]
                if variant.information_weighted_update:
                    weight = weight * selected_novelty[local_index]
                if label == 1:
                    positive_losses.append(
                        weight
                        * local_distillation(
                            policy, scenario, trace, positive=True
                        )
                    )
                elif label == -1:
                    negative_losses.append(
                        weight
                        * local_distillation(
                            policy, scenario, trace, positive=False
                        )
                    )
            if positive_losses:
                objective = objective + POSITIVE_WEIGHT * torch.mean(
                    torch.stack(positive_losses)
                )
            if negative_losses:
                objective = (
                    objective
                    + NEGATIVE_REJECTION_WEIGHT
                    * torch.mean(torch.stack(negative_losses))
                )
        else:
            scores = torch.stack(
                [robust_score(row) for row in teacher_matrix]
            )
            best_index = int(torch.argmin(scores))
            predicted_gain = scores[0] - scores[best_index]
            if best_index != 0 and float(predicted_gain) > threshold:
                weight = torch.clamp(
                    predicted_gain
                    / (0.10 + 0.02 * torch.abs(scores[0])),
                    0.0,
                    1.0,
                )
                objective = (
                    objective
                    + POSITIVE_WEIGHT
                    * weight
                    * global_positive_distillation(
                        policy, scenario, traces[best_index]
                    )
                )

        # Audit the exact policy state that produced the teacher labels.
        # Moving this check after optimizer.step() would compare labels from
        # two different policies and confound the reported accuracy.
        with torch.no_grad():
            audit_matrix, _ = branch_model_matrix(
                policy,
                scenario,
                selected_branches,
                selected_noises,
                config.train_horizon,
                [TRUE_MODEL],
                keep_traces=False,
            )
        audited_labels[episode, :selected_count] = (
            true_labels(audit_matrix, threshold).detach().cpu().numpy()
        )

        optimizer.zero_grad()
        optimizer_already_stepped = False
        if variant.balanced_conflict_safe_update:
            parameters = list(policy.parameters())
            trial_component = objective - baseline_objective
            base_gradients = torch.autograd.grad(
                baseline_objective,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            trial_gradients = torch.autograd.grad(
                trial_component,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            base_flat = _flatten_gradients(base_gradients, parameters)
            trial_flat = _flatten_gradients(trial_gradients, parameters)
            base_norm = torch.linalg.vector_norm(base_flat)
            trial_norm = torch.linalg.vector_norm(trial_flat)
            if float(trial_norm) > 1e-8:
                dot_product = torch.dot(base_flat, trial_flat)
                if float(dot_product) < 0.0 and float(base_norm) > 1e-8:
                    trial_flat = trial_flat - (
                        dot_product
                        / torch.clamp(base_norm**2, min=1e-8)
                    ) * base_flat
                    trial_norm = torch.linalg.vector_norm(trial_flat)
            combined = base_flat / torch.clamp(base_norm, min=1e-8)
            trial_weight = variant.trial_gradient_weight
            if variant.confidence_scaled_trial_update:
                accepted = labels != 0
                if bool(torch.any(accepted)):
                    accepted_fraction = torch.mean(accepted.float())
                    accepted_confidence = torch.mean(confidence[accepted])
                    trial_weight = trial_weight * float(
                        accepted_fraction * accepted_confidence
                    )
                else:
                    trial_weight = 0.0
            if variant.information_gated_trial_update:
                accepted = labels != 0
                if bool(torch.any(accepted)) and selected_count:
                    accepted_fraction = torch.mean(accepted.float())
                    accepted_confidence = torch.mean(confidence[accepted])
                    mean_novelty = torch.mean(selected_novelty)
                    reliability = torch.sqrt(
                        torch.clamp(
                            accepted_fraction
                            * accepted_confidence
                            * mean_novelty
                            * torch.mean(selection_confidence[selected]),
                            min=0.0,
                            max=1.0,
                        )
                    )
                    reliability_value = float(reliability)
                    if reliability_value < variant.minimum_trial_reliability:
                        trial_weight = 0.0
                    else:
                        trial_weight = trial_weight * reliability_value
                else:
                    trial_weight = 0.0
            if variant.trial_weight_half_life > 0.0:
                decay = 1.0 / math.sqrt(
                    1.0 + episode / variant.trial_weight_half_life
                )
                trial_weight = trial_weight * decay
            unit_trial = trial_flat / torch.clamp(trial_norm, min=1e-8)

            def combined_gradient(weight: float) -> tuple[torch.Tensor, torch.Tensor]:
                candidate = combined
                if float(trial_norm) > 1e-8:
                    candidate = candidate + weight * unit_trial
                candidate_norm = torch.linalg.vector_norm(candidate)
                candidate = candidate / torch.clamp(candidate_norm, min=1.0)
                return candidate, candidate_norm

            def assign_gradient(flat_gradient: torch.Tensor) -> None:
                offset = 0
                for parameter in parameters:
                    parameter_count = parameter.numel()
                    parameter.grad = flat_gradient[
                        offset : offset + parameter_count
                    ].reshape_as(parameter)
                    offset += parameter_count

            if variant.wm_safe_line_search and float(trial_norm) > 1e-8:
                candidate_weights = sorted(
                    {
                        0.0,
                        0.25 * trial_weight,
                        0.50 * trial_weight,
                        trial_weight,
                    }
                )
                base_policy_state = copy.deepcopy(policy.state_dict())
                base_optimizer_state = copy.deepcopy(optimizer.state_dict())
                candidate_scores: list[float] = []
                candidate_norms: list[torch.Tensor] = []
                for candidate_weight in candidate_weights:
                    policy.load_state_dict(base_policy_state)
                    optimizer.load_state_dict(base_optimizer_state)
                    optimizer.zero_grad()
                    candidate_gradient, candidate_norm = combined_gradient(
                        candidate_weight
                    )
                    assign_gradient(candidate_gradient)
                    optimizer.step()
                    candidate_scores.append(evaluate_wm_score())
                    candidate_norms.append(candidate_norm)
                    episode_wm_calls += wm_validation_call_count
                best_candidate = int(np.argmin(candidate_scores))
                base_score = candidate_scores[0]
                if (
                    base_score - candidate_scores[best_candidate]
                    < variant.line_search_margin
                ):
                    best_candidate = 0
                trial_weight = candidate_weights[best_candidate]
                wm_line_search_gain[episode] = (
                    base_score - candidate_scores[best_candidate]
                )
                policy.load_state_dict(base_policy_state)
                optimizer.load_state_dict(base_optimizer_state)
                optimizer.zero_grad()
                chosen_gradient, _ = combined_gradient(trial_weight)
                assign_gradient(chosen_gradient)
                optimizer.step()
                optimizer_already_stepped = True
                grad_norm = candidate_norms[best_candidate]
            else:
                chosen_gradient, combined_norm = combined_gradient(
                    trial_weight
                )
                assign_gradient(chosen_gradient)
                grad_norm = combined_norm
            effective_trial_weights[episode] = trial_weight
        else:
            objective.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=1.0
            )
        if not optimizer_already_stepped:
            optimizer.step()
        episode_wm_calls += update_wm_checkpoint(episode)
        objective_value = float(objective.detach())
        if not np.isfinite(objective_value):
            raise FloatingPointError(
                f"Non-finite objective for {variant.name}, episode {episode}."
            )
        training_objective[episode] = objective_value
        update_norm[episode] = float(grad_norm)
        model_rmse[episode + 1] = ensemble_rmse(models, holdout)
        wm_rollout_calls[episode] = episode_wm_calls

        if (
            episode % config.checkpoint_every == 0
            or episode == config.train_episodes - 1
        ):
            evaluation = deterministic_evaluation(
                policy, scenario, config.eval_horizon
            )
            checkpoint_episodes.append(episode + 1)
            checkpoint_steps.append(float(evaluation["steps"]))
            checkpoint_backlog.append(float(evaluation["backlog_area"]))

    if (
        variant.hf_checkpoint_selection
        and best_hf_policy_state is not None
    ):
        policy.load_state_dict(best_hf_policy_state)
    elif variant.wm_checkpoint_selection and best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
    wall_time = time.perf_counter() - start_time
    final = deterministic_evaluation(policy, scenario, config.eval_horizon)
    return {
        "policy": policy,
        "training_objective": training_objective,
        "nominal_objective": nominal_objective,
        "update_norm": update_norm,
        "gradient_logdet": gradient_logdet,
        "action_logdet": action_logdet,
        "positive_counts": positive_counts,
        "negative_counts": negative_counts,
        "ambiguous_counts": ambiguous_counts,
        "query_counts": query_counts,
        "high_fidelity_observation_counts": high_fidelity_observation_counts,
        "wm_rollout_calls": wm_rollout_calls,
        "selected_counts": selected_counts,
        "candidate_counts": candidate_counts,
        "selected_reliability": selected_reliability,
        "effective_trial_weights": effective_trial_weights,
        "wm_line_search_gain": wm_line_search_gain,
        "model_rmse": model_rmse,
        "predicted_labels": predicted_labels,
        "audited_labels": audited_labels,
        "selected_indices": selected_indices,
        "information_weights": information_weights,
        "checkpoint_episodes": np.asarray(checkpoint_episodes),
        "checkpoint_steps": np.asarray(checkpoint_steps),
        "checkpoint_backlog": np.asarray(checkpoint_backlog),
        "final_steps": float(final["steps"]),
        "final_completed": float(final["completed"]),
        "final_safe_completed": float(final["safe_completed"]),
        "final_collision": float(final["collision"]),
        "final_min_obstacle_clearance": float(
            final["min_obstacle_clearance"]
        ),
        "final_min_aav_separation": float(final["min_aav_separation"]),
        "final_backlog_area": float(final["backlog_area"]),
        "final_remaining": float(final["remaining"]),
        "best_wm_score": best_wm_score,
        "best_checkpoint_episode": float(best_checkpoint_episode),
        "best_hf_score": best_hf_score,
        "best_hf_checkpoint_episode": float(best_hf_checkpoint_episode),
        "trajectory": np.asarray(final["trajectory"]),
        "policy_vector": policy_vector(policy),
        "wall_time": wall_time,
        "query_buffer_size": float(len(calibration_buffer)),
    }


def label_statistics(
    predicted: np.ndarray, truth: np.ndarray
) -> tuple[float, float, int]:
    valid = truth != 2
    accepted = valid & (predicted != 0)
    accepted_count = int(np.sum(accepted))
    if accepted_count == 0:
        return float("nan"), 0.0, 0
    correct = accepted & (predicted == truth)
    return (
        float(np.sum(correct) / accepted_count),
        float(np.sum(correct) / max(int(np.sum(valid)), 1)),
        accepted_count,
    )


def _best(value: float, target: float) -> bool:
    return bool(np.isclose(value, target, rtol=1e-6, atol=1e-9))


def save_table(
    path: Path,
    steps: np.ndarray,
    success: np.ndarray,
    backlog: np.ndarray,
    correct_yield: np.ndarray,
    final_rmse: np.ndarray,
    wall_times: np.ndarray,
) -> None:
    med_steps = np.median(steps, axis=0)
    success_rate = 100.0 * np.mean(success, axis=0)
    med_backlog = np.median(backlog, axis=0)
    mean_yield = 100.0 * np.mean(correct_yield, axis=0)
    med_rmse = np.median(final_rmse, axis=0)
    med_time = np.median(wall_times, axis=0)
    best_values = (
        np.min(med_steps),
        np.max(success_rate),
        np.min(med_backlog),
        np.max(mean_yield),
        np.min(med_rmse),
        np.min(med_time),
    )
    lines = [
        "| Incremental variant | Completion slots $\\downarrow$ | "
        "Success $\\uparrow$ | Backlog area $\\downarrow$ | "
        "Correct labels/branch $\\uparrow$ | WM RMSE $\\downarrow$ | "
        "Training time (s) $\\downarrow$ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    columns = (
        med_steps,
        success_rate,
        med_backlog,
        mean_yield,
        med_rmse,
        med_time,
    )
    formats = ("{:.1f}", "{:.1f}%", "{:.2f}", "{:.1f}%", "{:.3f}", "{:.2f}")
    for method_index, variant in enumerate(VARIANTS):
        values = []
        for column, formatter, best_value in zip(
            columns, formats, best_values
        ):
            formatted = formatter.format(column[method_index])
            if _best(column[method_index], best_value):
                formatted = (
                    f'**<span style="color:#d62728">{formatted}</span>**'
                )
            values.append(formatted)
        lines.append(f"| {variant.name} | " + " | ".join(values) + " |")

    baseline = 0
    ours = len(VARIANTS) - 1
    completion_gain = med_steps[baseline] - med_steps[ours]
    backlog_gain = med_backlog[baseline] - med_backlog[ours]
    yield_gain = mean_yield[ours] - mean_yield[baseline]
    rmse_gain = med_rmse[baseline] - med_rmse[ours]
    lines.append(
        "| **Gain over action diversity** | "
        f"**{completion_gain:+.1f} slots** | "
        f"**{success_rate[ours] - success_rate[baseline]:+.1f} pp** | "
        f"**{backlog_gain:+.2f}** | "
        f"**{yield_gain:+.1f} pp** | "
        f"**{rmse_gain:+.3f}** | -- |"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(
    figure_dir: Path,
    steps: np.ndarray,
    backlog: np.ndarray,
    correct_yield: np.ndarray,
    gradient_logdet: np.ndarray,
    model_rmse: np.ndarray,
) -> None:
    rng = np.random.default_rng(20260730)
    positions = np.arange(1, len(VARIANTS) + 1)

    def distribution_plot(
        data: np.ndarray,
        ylabel: str,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        box = ax.boxplot(
            [data[:, index] for index in range(len(VARIANTS))],
            positions=positions,
            widths=0.60,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.8},
        )
        for patch, variant in zip(box["boxes"], VARIANTS):
            color = METHOD_COLORS[variant.name]
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.28)
        for index, variant in enumerate(VARIANTS):
            jitter = rng.normal(positions[index], 0.045, data.shape[0])
            ax.scatter(
                jitter,
                data[:, index],
                s=34,
                color=METHOD_COLORS[variant.name],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(SHORT_LABELS, rotation=22, ha="right", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.margins(x=0.02)
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(
                figure_dir / f"{filename}.{suffix}",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close(fig)

    distribution_plot(
        steps,
        "Completion time (slots)",
        "gradient_info_completion",
    )
    distribution_plot(
        backlog,
        "Backlog area",
        "gradient_info_backlog",
    )
    distribution_plot(
        100.0 * correct_yield,
        "Correct accepted labels per branch (%)",
        "gradient_info_label_yield",
    )
    distribution_plot(
        np.mean(gradient_logdet, axis=2),
        "Gradient-information log determinant",
        "gradient_info_logdet",
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    episodes = np.arange(model_rmse.shape[2])
    for method_index, variant in enumerate(VARIANTS):
        median = np.median(model_rmse[:, method_index], axis=0)
        lower = np.percentile(model_rmse[:, method_index], 25, axis=0)
        upper = np.percentile(model_rmse[:, method_index], 75, axis=0)
        ax.plot(
            episodes,
            median,
            color=METHOD_COLORS[variant.name],
            linewidth=2.0,
            label=variant.name,
        )
        ax.fill_between(
            episodes,
            lower,
            upper,
            color=METHOD_COLORS[variant.name],
            alpha=0.10,
            linewidth=0,
        )
    ax.set_xlabel("Policy-update episode", fontsize=14)
    ax.set_ylabel("Held-out WM rate RMSE", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"gradient_info_calibration.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def write_report(
    path: Path,
    config: ExperimentConfig,
    steps: np.ndarray,
    success: np.ndarray,
    backlog: np.ndarray,
    correct_yield: np.ndarray,
    final_rmse: np.ndarray,
    query_counts: np.ndarray,
) -> None:
    med_steps = np.median(steps, axis=0)
    success_rate = np.mean(success, axis=0)
    med_backlog = np.median(backlog, axis=0)
    mean_yield = np.mean(correct_yield, axis=0)
    med_rmse = np.median(final_rmse, axis=0)
    lines = [
        "# Gradient-Information World-Model Exploration",
        "",
        "## Protocol",
        "",
        f"- {config.n_trials} paired corridor layouts.",
        f"- {config.train_episodes} policy updates per variant.",
        f"- {config.pool_branches} candidate branches; "
        f"{config.selected_branches} retained per update.",
        "- Identical policy architecture, initialization, and optimizer across variants.",
        "- The high-fidelity model contains anisotropic attenuation that is outside "
        "the radial parametric ensemble family.",
        "- Only the final two variants receive at most one high-fidelity ambiguous-branch "
        "query per update.",
        "",
        "## Incremental results",
        "",
        "| Variant | Completion | Success | Backlog | Correct/branch | WM RMSE | Queries |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, variant in enumerate(VARIANTS):
        lines.append(
            f"| {variant.name} | {med_steps[index]:.1f} | "
            f"{100*success_rate[index]:.1f}% | {med_backlog[index]:.2f} | "
            f"{100*mean_yield[index]:.1f}% | {med_rmse[index]:.3f} | "
            f"{np.sum(query_counts[:, index]):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Gradient-information selection is supported only if its selected branches "
            "have a higher gradient log-determinant than action-diverse selection. "
            "Stable policy improvement additionally requires non-degraded success and "
            "backlog metrics.  Calibration is supported only if held-out rate RMSE "
            "decreases after high-fidelity queries.  These are small-scale mechanism "
            "checks, not paper-level statistical claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        return ExperimentConfig(1, 1, 10, 30, 3, 2, 1, "_smoke")
    if args.validate:
        return ExperimentConfig(2, 6, 25, 130, 5, 3, 2, "_validate")
    return ExperimentConfig(8, 20, 45, 220, 8, 4, 4, "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.validate:
        parser.error("--smoke and --validate are mutually exclusive")
    config = get_config(args)

    root = Path(__file__).resolve().parent
    result_dir = root / "results_extended" / "gradient_information_wm"
    raw_dir = result_dir / "raw"
    figure_dir = root / "paper_assets" / "figures"
    table_dir = root / "paper_assets" / "tables"
    for directory in (result_dir, raw_dir, figure_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    method_count = len(VARIANTS)
    base_shape = (config.n_trials, method_count)
    episode_shape = base_shape + (config.train_episodes,)
    final_steps = np.zeros(base_shape)
    final_success = np.zeros(base_shape)
    final_backlog = np.zeros(base_shape)
    final_remaining = np.zeros(base_shape)
    wall_times = np.zeros(base_shape)
    query_buffer_sizes = np.zeros(base_shape)
    objectives = np.zeros(episode_shape)
    nominal_objectives = np.zeros(episode_shape)
    update_norms = np.zeros(episode_shape)
    gradient_logdet = np.zeros(episode_shape)
    action_logdet = np.zeros(episode_shape)
    positive_counts = np.zeros(episode_shape)
    negative_counts = np.zeros(episode_shape)
    ambiguous_counts = np.zeros(episode_shape)
    query_counts = np.zeros(episode_shape)
    model_rmse = np.zeros(base_shape + (config.train_episodes + 1,))
    predicted_labels = np.zeros(
        episode_shape + (config.selected_branches,), dtype=np.int8
    )
    audited_labels = np.zeros_like(predicted_labels)
    correct_yield = np.zeros(base_shape)
    conditional_accuracy = np.zeros(base_shape)
    accepted_counts = np.zeros(base_shape)
    trajectories = np.empty(base_shape, dtype=object)
    policy_vectors = np.empty(base_shape, dtype=object)

    checkpoint_count = math.ceil(
        (config.train_episodes - 1) / config.checkpoint_every
    ) + 1
    convergence_steps = np.zeros(base_shape + (checkpoint_count,))
    convergence_backlog = np.zeros_like(convergence_steps)
    checkpoint_episodes: np.ndarray | None = None

    for trial in range(config.n_trials):
        scenario_seed = 83000 + trial
        initialization_seed = 93000 + trial
        scenario = make_validation_scenario(scenario_seed, num_users=4)
        raw_payload: dict[str, np.ndarray] = {
            "scenario_seed": np.asarray(scenario_seed),
            "initialization_seed": np.asarray(initialization_seed),
            "user_positions": scenario.user_positions.detach().cpu().numpy(),
            "user_tasks": scenario.tasks.detach().cpu().numpy(),
            "start_positions": scenario.start_positions.detach().cpu().numpy(),
        }
        for method_index, variant in enumerate(VARIANTS):
            result = train_variant(
                variant, scenario, config, initialization_seed
            )
            policy = result.pop("policy")
            final_steps[trial, method_index] = result["final_steps"]
            final_success[trial, method_index] = result["final_completed"]
            final_backlog[trial, method_index] = result["final_backlog_area"]
            final_remaining[trial, method_index] = result["final_remaining"]
            wall_times[trial, method_index] = result["wall_time"]
            query_buffer_sizes[trial, method_index] = result[
                "query_buffer_size"
            ]
            objectives[trial, method_index] = result["training_objective"]
            nominal_objectives[trial, method_index] = result[
                "nominal_objective"
            ]
            update_norms[trial, method_index] = result["update_norm"]
            gradient_logdet[trial, method_index] = result[
                "gradient_logdet"
            ]
            action_logdet[trial, method_index] = result["action_logdet"]
            positive_counts[trial, method_index] = result["positive_counts"]
            negative_counts[trial, method_index] = result["negative_counts"]
            ambiguous_counts[trial, method_index] = result[
                "ambiguous_counts"
            ]
            query_counts[trial, method_index] = result["query_counts"]
            model_rmse[trial, method_index] = result["model_rmse"]
            predicted_labels[trial, method_index] = result[
                "predicted_labels"
            ]
            audited_labels[trial, method_index] = result["audited_labels"]
            convergence_steps[trial, method_index] = result[
                "checkpoint_steps"
            ]
            convergence_backlog[trial, method_index] = result[
                "checkpoint_backlog"
            ]
            checkpoint_episodes = result["checkpoint_episodes"]
            trajectories[trial, method_index] = result["trajectory"]
            policy_vectors[trial, method_index] = result["policy_vector"]
            accuracy, yield_value, accepted = label_statistics(
                result["predicted_labels"], result["audited_labels"]
            )
            conditional_accuracy[trial, method_index] = accuracy
            correct_yield[trial, method_index] = yield_value
            accepted_counts[trial, method_index] = accepted

            token = (
                variant.name.lower()
                .replace(" ", "_")
                .replace("+", "plus")
                .replace("/", "n")
            )
            for key, value in result.items():
                raw_payload[f"{token}_{key}"] = np.asarray(value)
            print(
                f"trial {trial + 1}/{config.n_trials}, {variant.name}: "
                f"steps={final_steps[trial, method_index]:.0f}, "
                f"backlog={final_backlog[trial, method_index]:.2f}, "
                f"yield={100*yield_value:.1f}%, "
                f"rmse={model_rmse[trial, method_index, -1]:.3f}, "
                f"time={wall_times[trial, method_index]:.2f}s",
                flush=True,
            )
        raw_payload["variants"] = np.asarray(
            [variant.name for variant in VARIANTS]
        )
        raw_payload["metadata_json"] = np.asarray(
            json.dumps(
                {
                    **asdict(config),
                    "variants": [asdict(variant) for variant in VARIANTS],
                    "nominal_spec": asdict(NOMINAL_SPEC),
                    "manual_specs": [
                        asdict(spec) for spec in MANUAL_SPECS
                    ],
                    "channel_noise_std": CHANNEL_NOISE_STD,
                    "positive_weight": POSITIVE_WEIGHT,
                    "negative_rejection_weight": NEGATIVE_REJECTION_WEIGHT,
                    "device": str(DEVICE),
                }
            )
        )
        np.savez_compressed(
            raw_dir
            / f"gradient_information_wm{config.tag}_trial{trial + 1:02d}.npz",
            **raw_payload,
        )

    dense_policy_vectors = np.stack(
        [
            np.stack(
                [policy_vectors[trial, method] for method in range(method_count)]
            )
            for trial in range(config.n_trials)
        ]
    )
    aggregate = {
        "variants": np.asarray([variant.name for variant in VARIANTS]),
        "final_steps": final_steps,
        "final_success": final_success,
        "final_backlog_area": final_backlog,
        "final_remaining": final_remaining,
        "wall_times": wall_times,
        "query_buffer_sizes": query_buffer_sizes,
        "training_objectives": objectives,
        "nominal_objectives": nominal_objectives,
        "update_norms": update_norms,
        "gradient_logdet": gradient_logdet,
        "action_logdet": action_logdet,
        "positive_counts": positive_counts,
        "negative_counts": negative_counts,
        "ambiguous_counts": ambiguous_counts,
        "query_counts": query_counts,
        "model_rmse": model_rmse,
        "predicted_labels": predicted_labels,
        "audited_labels": audited_labels,
        "correct_label_yield": correct_yield,
        "conditional_label_accuracy": conditional_accuracy,
        "accepted_label_counts": accepted_counts,
        "convergence_steps": convergence_steps,
        "convergence_backlog_area": convergence_backlog,
        "checkpoint_episodes": checkpoint_episodes,
        "trajectories": trajectories,
        "policy_vectors": dense_policy_vectors,
        "metadata_json": np.asarray(
            json.dumps(
                {
                    **asdict(config),
                    "variants": [asdict(variant) for variant in VARIANTS],
                    "nominal_spec": asdict(NOMINAL_SPEC),
                    "manual_specs": [
                        asdict(spec) for spec in MANUAL_SPECS
                    ],
                    "channel_noise_std": CHANNEL_NOISE_STD,
                    "robustness_weight": ROBUSTNESS_WEIGHT,
                    "positive_weight": POSITIVE_WEIGHT,
                    "negative_rejection_weight": NEGATIVE_REJECTION_WEIGHT,
                    "local_rejection_margin": LOCAL_REJECTION_MARGIN,
                    "maximum_queries_per_episode": 1,
                    "device": str(DEVICE),
                }
            )
        ),
    }
    output_path = (
        result_dir / f"gradient_information_wm_validation{config.tag}.npz"
    )
    np.savez_compressed(output_path, **aggregate)
    save_table(
        table_dir / f"table_gradient_information_wm{config.tag}.md",
        final_steps,
        final_success,
        final_backlog,
        correct_yield,
        model_rmse[:, :, -1],
        wall_times,
    )
    plot_results(
        figure_dir,
        final_steps,
        final_backlog,
        correct_yield,
        gradient_logdet,
        model_rmse,
    )
    write_report(
        result_dir / f"validation_report{config.tag}.md",
        config,
        final_steps,
        final_success,
        final_backlog,
        correct_yield,
        model_rmse[:, :, -1],
        query_counts,
    )
    print(f"Saved {output_path}")
    if args.smoke:
        print("Smoke test PASSED")
    elif args.validate:
        print("Validation run PASSED")


if __name__ == "__main__":
    main()

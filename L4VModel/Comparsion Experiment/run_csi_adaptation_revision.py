"""Formal ablation of instantaneous-CSI feedback under slot-wise fading."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results_round2" / "csi_adaptation"


class CSIPolicy(nn.Module):
    """L4V policy with a dedicated per-user instantaneous-CSI input."""

    def __init__(self, num_users=4, max_step=0.2):
        super().__init__()
        self.num_users = num_users
        self.max_step = max_step
        input_dim = 2 + 2 * num_users + num_users + num_users
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.angle_head = nn.Linear(32, 1)
        self.speed_head = nn.Linear(32, 1)
        self.alloc_head = nn.Linear(32, num_users)
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in (self.angle_head, self.speed_head, self.alloc_head):
            nn.init.uniform_(layer.weight, -0.01, 0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, position, users, backlog, csi_feature):
        relative = users - position
        x = torch.cat(
            [position.flatten(), relative.flatten(), backlog, csi_feature]
        )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        angle = torch.tanh(self.angle_head(x)) * math.pi
        speed = torch.sigmoid(self.speed_head(x)) * self.max_step
        allocation = F.softmax(self.alloc_head(x), dim=-1)
        return angle, speed, allocation


def make_layout(seed, num_users=4, area_size=10.0):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    users = (
        torch.rand((num_users, 2), generator=generator, device=DEVICE) - 0.5
    ) * area_size
    tasks = 2.0 + 2.0 * torch.rand(
        num_users, generator=generator, device=DEVICE
    )
    return users, tasks


def fading_multiplier(num_users, generator, shadow_std=0.30, rician_k=2.0):
    shadow = torch.exp(
        shadow_std
        * torch.randn(num_users, generator=generator, device=DEVICE)
    )
    los_amp = math.sqrt(rician_k / (rician_k + 1.0))
    scatter_std = math.sqrt(1.0 / (2.0 * (rician_k + 1.0)))
    eps_r = torch.randn(num_users, generator=generator, device=DEVICE)
    eps_i = torch.randn(num_users, generator=generator, device=DEVICE)
    rician = (los_amp + scatter_std * eps_r) ** 2 + (
        scatter_std * eps_i
    ) ** 2
    return shadow * rician


def rates_from_multiplier(position, users, multiplier, sigma=0.1):
    distance_sq = torch.sum((position - users) ** 2, dim=1).clamp(min=0.1)
    return torch.log1p(multiplier / (sigma * distance_sq))


def advance(position, angle, speed):
    movement = torch.stack(
        [torch.cos(angle) * speed, torch.sin(angle) * speed], dim=-1
    ).view_as(position)
    return position + movement


def csi_feature(multiplier, enabled):
    if not enabled:
        return torch.zeros_like(multiplier)
    return torch.clamp(torch.log(multiplier + 1e-8), min=-3.0, max=3.0)


def train_policy(
    users,
    tasks,
    seed,
    use_current_csi,
    updates,
    horizon,
):
    torch.manual_seed(seed)
    policy = CSIPolicy(num_users=users.shape[0]).to(DEVICE)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    losses = np.zeros(updates, dtype=np.float64)
    started = time.perf_counter()
    for update in range(updates):
        channel_generator = torch.Generator(device=DEVICE)
        channel_generator.manual_seed(seed * 100_000 + update)
        position = torch.zeros((1, 2), device=DEVICE)
        backlog = tasks.clone()
        initial_total = torch.sum(tasks)
        objective = torch.zeros((), device=DEVICE)
        angles = []
        for _ in range(horizon):
            multiplier = fading_multiplier(users.shape[0], channel_generator)
            angle, speed, allocation = policy(
                position,
                users,
                backlog,
                csi_feature(multiplier, use_current_csi),
            )
            position = advance(position, angle, speed)
            rates = rates_from_multiplier(position, users, multiplier)
            backlog = torch.clamp(
                backlog - allocation * users.shape[0] * rates * 0.1,
                min=0.0,
            )
            objective = objective + torch.sum(backlog) / initial_total
            angles.append(angle)
            if float(torch.sum(backlog).detach()) < 1e-3 * users.shape[0]:
                break
        if len(angles) > 1:
            objective = objective + 0.01 * torch.mean(
                torch.diff(torch.stack(angles).flatten()) ** 2
            )
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses[update] = float(objective.detach())
        if (update + 1) % max(1, updates // 5) == 0:
            print(
                f"seed={seed} csi={int(use_current_csi)} "
                f"update={update + 1}/{updates} loss={losses[update]:.3f}",
                flush=True,
            )
    return policy, losses, time.perf_counter() - started


def evaluate_policy(
    policy,
    users,
    tasks,
    seed,
    use_current_csi,
    eval_rollouts,
    horizon,
):
    completion = np.full(eval_rollouts, horizon, dtype=np.float64)
    average_completion = np.full(eval_rollouts, horizon, dtype=np.float64)
    success = np.zeros(eval_rollouts, dtype=bool)
    correlations = np.zeros(eval_rollouts, dtype=np.float64)
    best_user_share = np.zeros(eval_rollouts, dtype=np.float64)
    action_sensitivity = np.zeros(eval_rollouts, dtype=np.float64)
    inference_ms = np.zeros(eval_rollouts, dtype=np.float64)
    policy.eval()
    for rollout in range(eval_rollouts + 5):
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed * 1_000_000 + 500_000 + rollout)
        position = torch.zeros((1, 2), device=DEVICE)
        backlog = tasks.clone()
        completed = torch.zeros(users.shape[0], dtype=torch.bool, device=DEVICE)
        per_user_completion = torch.full(
            (users.shape[0],), horizon, dtype=torch.float32, device=DEVICE
        )
        all_allocations = []
        all_log_csi = []
        best_shares = []
        sensitivities = []
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            for step in range(horizon):
                multiplier = fading_multiplier(users.shape[0], generator)
                feature = csi_feature(multiplier, use_current_csi)
                angle, speed, allocation = policy(
                    position, users, backlog, feature
                )
                permuted = torch.roll(feature, shifts=1, dims=0)
                _, _, allocation_permuted = policy(
                    position, users, backlog, permuted
                )
                sensitivities.append(
                    float(torch.mean(torch.abs(allocation - allocation_permuted)))
                )
                active = backlog > 1e-8
                if bool(torch.any(active)):
                    all_allocations.extend(
                        allocation[active].detach().cpu().tolist()
                    )
                    all_log_csi.extend(
                        torch.log(multiplier[active] + 1e-8)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    best = torch.argmax(
                        torch.where(
                            active,
                            multiplier,
                            torch.full_like(multiplier, -1.0),
                        )
                    )
                    best_shares.append(float(allocation[best]))
                position = advance(position, angle, speed)
                rates = rates_from_multiplier(position, users, multiplier)
                backlog = torch.clamp(
                    backlog - allocation * users.shape[0] * rates * 0.1,
                    min=0.0,
                )
                newly_completed = (~completed) & (backlog <= 1e-8)
                per_user_completion[newly_completed] = step + 1
                completed = completed | newly_completed
                if bool(torch.all(completed)):
                    break
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = 1000.0 * (time.perf_counter() - started)
        if rollout < 5:
            continue
        index = rollout - 5
        success[index] = bool(torch.all(completed))
        if success[index]:
            completion[index] = step + 1
        average_completion[index] = float(torch.mean(per_user_completion))
        if len(all_allocations) > 1 and np.std(all_log_csi) > 0:
            correlations[index] = float(
                np.corrcoef(all_allocations, all_log_csi)[0, 1]
            )
        best_user_share[index] = float(np.mean(best_shares))
        action_sensitivity[index] = float(np.mean(sensitivities))
        inference_ms[index] = elapsed
    return {
        "completion_slots": completion,
        "average_completion_slots": average_completion,
        "success": success,
        "allocation_log_csi_correlation": correlations,
        "best_csi_user_allocation_share": best_user_share,
        "allocation_permutation_sensitivity": action_sensitivity,
        "direct_inference_ms": inference_ms,
    }


def run(smoke=False):
    trials = 1 if smoke else 8
    updates = 10 if smoke else 500
    train_horizon = 40 if smoke else 160
    eval_rollouts = 3 if smoke else 20
    eval_horizon = 60 if smoke else 250
    tag = "_smoke" if smoke else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records_path = OUTPUT_DIR / f"records{tag}.jsonl"
    records_path.write_text("", encoding="utf-8")
    arrays = {}
    records = []
    for trial in range(trials):
        layout_seed = 40930000 + trial
        users, tasks = make_layout(layout_seed)
        for method, enabled in (("backlog_only", False), ("current_csi", True)):
            policy, losses, train_time = train_policy(
                users,
                tasks,
                layout_seed + 1000,
                enabled,
                updates,
                train_horizon,
            )
            evaluation = evaluate_policy(
                policy,
                users,
                tasks,
                layout_seed + 2000,
                enabled,
                eval_rollouts,
                eval_horizon,
            )
            prefix = f"{method}_trial{trial + 1:02d}"
            arrays[f"{prefix}_losses"] = losses
            for key, value in evaluation.items():
                arrays[f"{prefix}_{key}"] = value
            arrays[f"{prefix}_users"] = users.detach().cpu().numpy()
            arrays[f"{prefix}_tasks"] = tasks.detach().cpu().numpy()
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "layout_seed": layout_seed,
                    "method": method,
                    "users": users.detach().cpu(),
                    "tasks": tasks.detach().cpu(),
                },
                checkpoint_dir / f"{prefix}.pt",
            )
            record = {
                "trial": trial,
                "layout_seed": layout_seed,
                "method": method,
                "training_time_s": train_time,
                "completion_fraction": float(np.mean(evaluation["success"])),
                "median_completion_slots": float(
                    np.median(evaluation["completion_slots"])
                ),
                "median_average_completion_slots": float(
                    np.median(evaluation["average_completion_slots"])
                ),
                "median_allocation_log_csi_correlation": float(
                    np.median(evaluation["allocation_log_csi_correlation"])
                ),
                "median_best_csi_user_allocation_share": float(
                    np.median(evaluation["best_csi_user_allocation_share"])
                ),
                "median_allocation_permutation_sensitivity": float(
                    np.median(evaluation["allocation_permutation_sensitivity"])
                ),
                "median_direct_inference_ms": float(
                    np.median(evaluation["direct_inference_ms"])
                ),
            }
            records.append(record)
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

    output = OUTPUT_DIR / f"csi_adaptation{tag}.npz"
    np.savez_compressed(
        output,
        trials=np.asarray(trials),
        updates=np.asarray(updates),
        train_horizon=np.asarray(train_horizon),
        eval_rollouts=np.asarray(eval_rollouts),
        eval_horizon=np.asarray(eval_horizon),
        shadow_std=np.asarray(0.30),
        rician_k=np.asarray(2.0),
        **arrays,
    )
    summary = {
        "protocol": "round2_csi_adaptation_smoke" if smoke else "round2_csi_adaptation_formal",
        "paired_layouts": trials,
        "evaluation_rollouts_per_layout": eval_rollouts,
        "updates_per_policy": updates,
        "shadow_std": 0.30,
        "rician_k": 2.0,
        "methods": {},
    }
    for method in ("backlog_only", "current_csi"):
        selected = [record for record in records if record["method"] == method]
        summary["methods"][method] = {
            key: float(np.median([record[key] for record in selected]))
            for key in (
                "completion_fraction",
                "median_completion_slots",
                "median_average_completion_slots",
                "median_allocation_log_csi_correlation",
                "median_best_csi_user_allocation_share",
                "median_allocation_permutation_sensitivity",
                "median_direct_inference_ms",
                "training_time_s",
            )
        }
    summary_path = OUTPUT_DIR / f"summary{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.smoke)

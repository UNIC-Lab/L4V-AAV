"""Successive-convex-approximation QCQP baseline for L4V revision round 2.

The baseline searches for the shortest feasible mission horizon under the
deterministic equal-split channel used in the paper's broad sweeps.  At every
SCA iteration, the convex rate function in squared distance is replaced by its
global affine lower bound.  Motion and squared-distance epigraph constraints
then form a convex QCQP.  The saved result includes the true nonlinear rollout,
solver status, iteration counts, and wall-clock cost.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import torch


@dataclass
class QCQPResult:
    path: np.ndarray
    completion_times: np.ndarray
    mission_time: int
    average_completion_time: float
    solve_time_s: float
    horizon: int
    sca_iterations: int
    status: str
    max_service_deficit: float


def make_instance(seed: int, num_users: int = 4, area_size: float = 10.0):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    users = (torch.rand((num_users, 2), generator=generator) - 0.5) * area_size
    tasks = torch.rand(num_users, generator=generator) * 2.0 + 2.0
    return users.numpy().astype(np.float64), tasks.numpy().astype(np.float64)


def rate_from_squared_distance(z, h_unit=1.0, sigma=0.1):
    z = np.maximum(np.asarray(z, dtype=np.float64), 0.1)
    return np.log1p(float(h_unit) / (float(sigma) * z))


def _nearest_neighbor_reference(users, horizon, max_step):
    current = np.zeros(2, dtype=np.float64)
    remaining = list(range(len(users)))
    waypoints = [current.copy()]
    while remaining:
        index = min(remaining, key=lambda i: np.linalg.norm(users[i] - current))
        target = users[index]
        delta = target - current
        steps = max(1, int(math.ceil(np.linalg.norm(delta) / max_step)))
        for step in range(1, steps + 1):
            waypoints.append(current + delta * step / steps)
        current = target.copy()
        remaining.remove(index)
    path = np.asarray(waypoints)
    if len(path) < horizon + 1:
        path = np.vstack([path, np.repeat(path[-1][None], horizon + 1 - len(path), axis=0)])
    return path[: horizon + 1]


def _rollout(path, users, tasks, time_step=0.1, h_unit=1.0, sigma=0.1):
    distance2 = np.sum((path[1:, None, :] - users[None, :, :]) ** 2, axis=-1)
    service = rate_from_squared_distance(distance2, h_unit, sigma) * float(time_step)
    cumulative = np.cumsum(service, axis=0)
    completion = np.full(len(tasks), len(path), dtype=np.int64)
    for user in range(len(tasks)):
        reached = np.flatnonzero(cumulative[:, user] >= tasks[user] - 1e-7)
        if len(reached):
            completion[user] = int(reached[0] + 1)
    delivered = cumulative[-1]
    return completion, delivered


def solve_horizon(users, tasks, horizon, *, max_step=0.2, time_step=0.1,
                  h_unit=1.0, sigma=0.1, area_size=10.0, sca_iters=6,
                  solver="CLARABEL"):
    users = np.asarray(users, dtype=np.float64)
    tasks = np.asarray(tasks, dtype=np.float64)
    reference = _nearest_neighbor_reference(users, horizon, max_step)
    started = time.perf_counter()
    last_status = "not_run"
    used_iterations = 0

    for iteration in range(int(sca_iters)):
        position = cp.Variable((horizon + 1, 2))
        squared_distance = cp.Variable((horizon, len(users)))
        slack = cp.Variable(len(users), nonneg=True)
        constraints = [position[0] == 0.0]
        for step in range(horizon):
            constraints.append(cp.sum_squares(position[step + 1] - position[step]) <= max_step ** 2)
        constraints += [position <= area_size / 2, position >= -area_size / 2]

        lower_rates = []
        for user in range(len(users)):
            terms = []
            for step in range(horizon):
                z0 = max(float(np.sum((reference[step + 1] - users[user]) ** 2)), 0.1)
                a = float(h_unit) / float(sigma)
                value = math.log1p(a / z0)
                derivative = -a / (z0 * (z0 + a))
                constraints += [
                    squared_distance[step, user] >= cp.sum_squares(position[step + 1] - users[user]),
                    squared_distance[step, user] >= 0.1,
                ]
                terms.append(value + derivative * (squared_distance[step, user] - z0))
            lower_rates.append(cp.sum(cp.hstack(terms)) * time_step)
            constraints.append(lower_rates[-1] + slack[user] >= tasks[user])

        objective = cp.Minimize(1e5 * cp.sum(slack) + cp.sum_squares(position - reference))
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            problem.solve(solver="SCS", eps=1e-5, max_iters=20000, verbose=False)
        last_status = str(problem.status)
        if position.value is None:
            break
        candidate = np.asarray(position.value, dtype=np.float64)
        used_iterations = iteration + 1
        change = float(np.max(np.linalg.norm(candidate - reference, axis=1)))
        reference = candidate
        if change < 1e-3:
            break

    elapsed = time.perf_counter() - started
    completion, delivered = _rollout(reference, users, tasks, time_step, h_unit, sigma)
    deficit = float(np.max(np.maximum(tasks - delivered, 0.0)))
    feasible = bool(np.all(completion <= horizon) and deficit <= 1e-4)
    return {
        "feasible": feasible,
        "path": reference,
        "completion": completion,
        "solve_time_s": elapsed,
        "horizon": int(horizon),
        "sca_iterations": used_iterations,
        "status": last_status,
        "max_service_deficit": deficit,
    }


def qcqp_optimize(*, seed, num_users=4, area_size=10.0, h_unit=1.0,
                   sigma=0.1, max_step=0.2, time_step=0.1,
                   max_horizon=160, sca_iters=6):
    users, tasks = make_instance(seed, num_users, area_size)
    total_started = time.perf_counter()
    attempts = []
    lower = 4
    upper = 10
    solved = None
    while upper <= max_horizon:
        trial = solve_horizon(
            users, tasks, upper, max_step=max_step, time_step=time_step,
            h_unit=h_unit, sigma=sigma, area_size=area_size, sca_iters=sca_iters,
        )
        attempts.append(trial)
        if trial["feasible"]:
            solved = trial
            break
        lower = upper + 1
        upper = min(max_horizon, int(math.ceil(upper * 1.5)))
        if upper < lower:
            break
    if solved is None:
        trial = solve_horizon(
            users, tasks, max_horizon, max_step=max_step, time_step=time_step,
            h_unit=h_unit, sigma=sigma, area_size=area_size, sca_iters=sca_iters,
        )
        attempts.append(trial)
        solved = trial
    elif lower < solved["horizon"]:
        left, right = lower, solved["horizon"]
        while left < right:
            middle = (left + right) // 2
            trial = solve_horizon(
                users, tasks, middle, max_step=max_step, time_step=time_step,
                h_unit=h_unit, sigma=sigma, area_size=area_size, sca_iters=sca_iters,
            )
            attempts.append(trial)
            if trial["feasible"]:
                solved = trial
                right = middle
            else:
                left = middle + 1

    completion = np.asarray(solved["completion"], dtype=np.int64)
    mission = int(np.max(completion))
    return QCQPResult(
        path=np.asarray(solved["path"]),
        completion_times=completion,
        mission_time=mission,
        average_completion_time=float(np.mean(completion)),
        solve_time_s=float(time.perf_counter() - total_started),
        horizon=int(solved["horizon"]),
        sca_iterations=int(sum(item["sca_iterations"] for item in attempts)),
        status=str(solved["status"]),
        max_service_deficit=float(solved["max_service_deficit"]),
    ), users, tasks


__all__ = ["QCQPResult", "make_instance", "qcqp_optimize", "solve_horizon"]

"""
mpc_optimization_extended.py

Differentiable receding-horizon model predictive control (MPC) baseline,
requested by Reviewer 1 ("additional comparisons with model predictive
control (MPC), optimal control methods ... would provide a more comprehensive
evaluation"). Unlike L4V, this baseline has no learned policy: at every real
time step it re-optimizes a short open-loop control sequence directly by
gradient descent on the same differentiable simulator used by L4V, executes
only the first action, then discards and re-solves at the next step (shooting
method / receding horizon control). This makes it the most apples-to-apples
optimal-control comparison available, since both methods differentiate through
the identical dynamics and channel model.

Because MPC re-optimizes online rather than paying a one-off training cost, its
'total_training_time' here reports the full per-mission online compute time,
which is the quantity that matters for a real-time deployment comparison
against L4V's near-zero-cost inference after training.
"""
import time
import math
import torch
import torch.nn.functional as F

from env_extended import (
    device,
    initialize_environment,
    g,
    f_extended,
    propulsion_power,
)


def mpc_optimize(num_users, area_size, h_unit, sigma, max_step, time_step_duration,
                  horizon=15, inner_iters=30, inner_lr=0.05, max_episodes=500,
                  use_allocation=True, shadow_std=0.0, rician_k=None,
                  v_scale=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    user_positions, initial_tasks, base_station_pos = initialize_environment(num_users, area_size)
    h_unit_t = torch.tensor(float(h_unit), device=device)
    sigma_t = torch.tensor(float(sigma), device=device)

    base_pos = base_station_pos.clone()
    remaining_tasks = initial_tasks.clone()

    v_window = torch.zeros(horizon, device=device)
    theta_window = torch.rand(horizon, device=device) * 2 * math.pi
    alloc_logits_window = torch.zeros(horizon, num_users, device=device)

    transmission_rates = []
    total_energy = 0.0
    time_steps = 0

    start = time.time()
    for t in range(max_episodes):
        v_var = v_window.clone().detach().requires_grad_(True)
        theta_var = theta_window.clone().detach().requires_grad_(True)
        alloc_var = alloc_logits_window.clone().detach().requires_grad_(True)
        inner_opt = torch.optim.Adam([v_var, theta_var, alloc_var], lr=inner_lr)

        for _ in range(inner_iters):
            inner_opt.zero_grad()
            pos_sim = base_pos.clone()
            tasks_sim = remaining_tasks.clone()
            rollout_loss = 0.0
            for h in range(horizon):
                speed_h = torch.sigmoid(v_var[h]) * max_step
                angle_h = torch.tanh(theta_var[h]) * math.pi
                if use_allocation:
                    alloc_h = F.softmax(alloc_var[h], dim=-1)
                else:
                    alloc_h = torch.full((num_users,), 1.0 / num_users, device=device)

                pos_sim = g(pos_sim, angle_h, speed_h)
                step_loss, tasks_sim, _ = f_extended(
                    pos_sim, user_positions, tasks_sim, alloc_h, h_unit_t, sigma_t,
                    time_step_duration, area_size, shadow_std=shadow_std, rician_k=rician_k,
                )
                rollout_loss = rollout_loss + step_loss
            rollout_loss.backward()
            inner_opt.step()

        with torch.no_grad():
            speed0 = torch.sigmoid(v_var[0]) * max_step
            angle0 = torch.tanh(theta_var[0]) * math.pi
            if use_allocation:
                alloc0 = F.softmax(alloc_var[0], dim=-1)
            else:
                alloc0 = torch.full((num_users,), 1.0 / num_users, device=device)

            base_pos = g(base_pos, angle0, speed0)
            step_loss, remaining_tasks, rates = f_extended(
                base_pos, user_positions, remaining_tasks, alloc0, h_unit_t, sigma_t,
                time_step_duration, area_size, shadow_std=shadow_std, rician_k=rician_k,
            )
            transmission_rates.extend(rates.tolist())
            total_energy += float(propulsion_power(speed0, v_scale=v_scale).item()) * time_step_duration

        # Warm-start the next receding-horizon solve by shifting the window.
        v_window = torch.cat([v_var[1:], v_var[-1:]]).detach()
        theta_window = torch.cat([theta_var[1:], theta_var[-1:]]).detach()
        alloc_logits_window = torch.cat([alloc_var[1:], alloc_var[-1:]], dim=0).detach()

        time_steps += 1
        if torch.sum(remaining_tasks) < 0.001 * num_users:
            break

    total_time = time.time() - start
    total_tasks = torch.sum(initial_tasks).item()
    avg_task_reduction = total_tasks / time_steps if time_steps > 0 else 0

    return {
        'time_steps': time_steps,
        'convergence_episode': float('nan'),
        'convergence_time': total_time,
        'total_training_time': total_time,
        'avg_task_reduction': avg_task_reduction,
        'transmission_rates': transmission_rates,
        'total_energy': total_energy,
    }

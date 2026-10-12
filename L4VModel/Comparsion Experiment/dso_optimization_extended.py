"""
dso_optimization_extended.py

L4V training on top of env_extended.py: adds a learned OFDMA allocation head,
a reparameterized stochastic (shadowed / Rician-faded) channel, and an optional
energy-weighted loss term using the Zeng, Xu & Zhang rotary-wing power model.

Kept separate from dso_optimization.py so the paper's originally reported
static-channel / equal-split results stay reproducible unchanged.
"""
import time
import torch
import torch.nn.functional as F

from env_extended import (
    device,
    initialize_environment,
    compute_transmission_rate_stochastic,
    g,
    f_extended,
    propulsion_power,
    PathPlanningNetExtended,
)


def dso_optimize_extended(num_users, area_size, h_unit, sigma, max_step, time_step_duration,
                           convergence_threshold=1e-3, max_episodes=2000,
                           use_allocation=True, shadow_std=0.0, rician_k=None,
                           energy_weight=0.0, v_scale=1.0, seed=None,
                           user_positions_override=None, initial_tasks_override=None,
                           exact_evaluation=False, return_model=False):
    """Same protocol/metrics as dso_optimize, extended with allocation/stochastic
    channel/energy. use_allocation=False and shadow_std=0/rician_k=None recovers
    the original static equal-split, deterministic-channel model exactly."""
    if seed is not None:
        torch.manual_seed(seed)

    if (user_positions_override is None) != (initial_tasks_override is None):
        raise ValueError("Both instance overrides must be provided together")
    if user_positions_override is None:
        user_positions, initial_tasks, base_station_pos = initialize_environment(num_users, area_size)
    else:
        user_positions = torch.as_tensor(user_positions_override, dtype=torch.float32, device=device)
        initial_tasks = torch.as_tensor(initial_tasks_override, dtype=torch.float32, device=device)
        if user_positions.shape != (num_users, 2) or initial_tasks.shape != (num_users,):
            raise ValueError("Instance override shape does not match num_users")
        base_station_pos = torch.zeros((1, 2), dtype=torch.float32, device=device)
    h_unit_t = torch.tensor(float(h_unit), device=device)
    sigma_t = torch.tensor(float(sigma), device=device)

    model = PathPlanningNetExtended(num_users, v_max=max_step).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

    trajectory_length = 500
    lambda_preact = 1e-3

    loss_history = []
    convergence_episode = None
    convergence_time = None
    prev_loss = float('inf')
    stable_count = 0

    start = time.time()
    for episode in range(max_episodes):
        model.train()
        base_pos = base_station_pos.clone()
        remaining_tasks = initial_tasks.clone()

        total_loss = 0.0
        l2_preact_reg = 0.0
        energy_accum = 0.0

        angles_seq = []
        speeds_seq = []

        for t in range(trajectory_length):
            angle, speed, alloc, angle_preact = model(base_pos, user_positions, remaining_tasks,
                                                        use_allocation=use_allocation)
            l2_preact_reg = l2_preact_reg + torch.mean(angle_preact ** 2)

            base_pos = g(base_pos, angle, speed)

            step_loss, remaining_tasks, _ = f_extended(
                base_pos, user_positions, remaining_tasks, alloc, h_unit_t, sigma_t,
                time_step_duration, area_size, shadow_std=shadow_std, rician_k=rician_k,
            )
            total_loss = total_loss + step_loss / torch.sum(initial_tasks)

            if energy_weight > 0:
                energy_accum = energy_accum + propulsion_power(speed, v_scale=v_scale) * time_step_duration

            angles_seq.append(angle)
            speeds_seq.append(speed)

            if torch.sum(remaining_tasks) < 0.001 * num_users:
                break

        if len(angles_seq) > 1:
            angles_tensor = torch.stack(angles_seq).squeeze()
            angle_diff = torch.diff(angles_tensor)
            smoothness_penalty = 0.01 * torch.mean(angle_diff ** 2)
            total_loss = total_loss + smoothness_penalty

        total_loss = total_loss + lambda_preact * l2_preact_reg
        if energy_weight > 0:
            total_loss = total_loss + energy_weight * energy_accum

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss = float(total_loss.item())
        loss_history.append(current_loss)

        if abs(prev_loss - current_loss) < convergence_threshold:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= 5 and convergence_episode is None:
            convergence_episode = episode - 4
            convergence_time = time.time() - start
            break

        prev_loss = current_loss
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {prev_loss:.4f}, TotalTime: {time.time() - start:.4f}")

        if episode > 100 and current_loss < 0.01:
            break

    total_training_time = time.time() - start

    # Evaluation rollout with the trained (deterministic) policy.
    with torch.no_grad():
        base_pos = base_station_pos.clone()
        remaining_tasks = initial_tasks.clone()
        time_steps = 0
        transmission_rates = []
        total_energy = 0.0
        speed_sequence = []
        trajectory = [base_pos.detach().cpu().numpy().reshape(2)]
        completion_times = torch.full(
            (num_users,), trajectory_length, dtype=torch.int64, device=device
        )
        completed = torch.zeros(num_users, dtype=torch.bool, device=device)

        for t in range(trajectory_length):
            angle, speed, alloc, _ = model(base_pos, user_positions, remaining_tasks,
                                            use_allocation=use_allocation)
            base_pos = g(base_pos, angle, speed)

            rates = compute_transmission_rate_stochastic(
                base_pos, user_positions, h_unit_t, sigma_t,
                shadow_std=shadow_std, rician_k=rician_k,
            )
            transmitted = alloc * num_users * rates * time_step_duration
            remaining_tasks = torch.clamp(remaining_tasks - transmitted, min=0)
            newly_completed = (~completed) & (remaining_tasks <= 1e-8)
            completion_times[newly_completed] = t + 1
            completed = completed | newly_completed
            total_energy += float(propulsion_power(speed, v_scale=v_scale).item()) * time_step_duration

            transmission_rates.extend(rates.tolist())
            speed_sequence.append(float(speed.item()))
            trajectory.append(base_pos.detach().cpu().numpy().reshape(2))
            time_steps += 1

            if exact_evaluation:
                finished = bool(torch.all(completed))
            else:
                finished = bool(torch.sum(remaining_tasks) < 0.001 * num_users)
                if finished:
                    completion_times[~completed] = t + 1
            if finished:
                break

        total_tasks = torch.sum(initial_tasks).item()
        avg_task_reduction = total_tasks / time_steps if time_steps > 0 else 0

    if convergence_episode is None:
        convergence_episode = max_episodes
        convergence_time = total_training_time

    result = {
        'time_steps': time_steps,
        'per_user_completion_times': completion_times.detach().cpu().numpy(),
        'average_completion_time': float(completion_times.float().mean().item()),
        'convergence_episode': convergence_episode,
        'convergence_time': convergence_time,
        'total_training_time': total_training_time,
        'avg_task_reduction': avg_task_reduction,
        'transmission_rates': transmission_rates,
        'total_energy': total_energy,
        'speed_sequence': speed_sequence,
        'trajectory': trajectory,
    }
    if return_model:
        result['model'] = model
    return result

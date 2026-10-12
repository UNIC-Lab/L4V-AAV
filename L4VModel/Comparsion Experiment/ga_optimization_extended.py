"""GA baseline under the same reparameterized stochastic channel as the other
extended baselines (log-normal shadowing + Rician fading, matching
stochastic_patch.py's shadow_std=0.15, rician_k=6.0). Pure-Python/numpy
reimplementation of the channel term since ga_optimization.py has no torch
dependency.

Also fixes a latent bug in the original eval_fitness_improved: it computed
`rate = log2(1+h)` with sigma never applied during evolutionary search (only
analyze_solution used sigma, for reporting only), so the original GA optimizes
blind to noise power entirely. For this stochastic-channel variant, where the
whole point is testing robustness to noise, the fitness function must actually
see sigma and the fading realization, so both are included here. The original
ga_optimization.py is left untouched so the paper's originally reported GA
numbers stay reproducible.
"""
import math
import random

import ga_optimization as _ga

DEFAULT_SHADOW_STD = 0.15
DEFAULT_RICIAN_K = 6.0


def _fading_gain(shadow_std, rician_k):
    shadow = math.exp(shadow_std * random.gauss(0, 1))
    los_amp = math.sqrt(rician_k / (rician_k + 1.0))
    nlos_std = math.sqrt(1.0 / (2.0 * (rician_k + 1.0)))
    er, ei = random.gauss(0, 1), random.gauss(0, 1)
    fading = (los_amp + nlos_std * er) ** 2 + (nlos_std * ei) ** 2
    return shadow * fading


def _make_eval_fitness_stochastic(shadow_std, rician_k):
    def eval_fitness_stochastic(individual, user_positions, user_tasks, h0, sigma,
                                 time_step_duration, num_time_steps):
        positions = []
        current_x, current_y = 0.0, 0.0
        positions.append((current_x, current_y))
        for dx, dy in individual:
            current_x += dx
            current_y += dy
            positions.append((current_x, current_y))

        remaining_data = user_tasks.copy()
        total_time_steps_needed = num_time_steps

        for t in range(num_time_steps):
            pos = positions[t]
            for i in range(len(user_positions)):
                if remaining_data[i] <= 0:
                    continue
                user_pos = user_positions[i]
                distance = math.sqrt((pos[0] - user_pos[0]) ** 2 + (pos[1] - user_pos[1]) ** 2)
                if distance < 0.001:
                    distance = 0.001

                h = h0 / (distance ** 2) * _fading_gain(shadow_std, rician_k)
                rate = math.log(1 + h / sigma)
                data_transmitted = rate * time_step_duration

                if data_transmitted > remaining_data[i]:
                    data_transmitted = remaining_data[i]
                remaining_data[i] -= data_transmitted

            if all(d <= 0.001 * len(user_positions) for d in remaining_data):
                total_time_steps_needed = t + 1
                break

        penalty = 0
        unfinished_data = sum(max(0, d) for d in remaining_data)
        if unfinished_data > 0.001 * len(user_positions):
            penalty = unfinished_data * 20000

        total_time = total_time_steps_needed ** 2 + penalty
        return total_time,

    return eval_fitness_stochastic


def _make_analyze_solution_stochastic(shadow_std, rician_k):
    def analyze_solution_stochastic(individual, user_positions, user_tasks, h0, sigma,
                                     time_step_duration, num_time_steps):
        positions = []
        current_x, current_y = 0.0, 0.0
        positions.append((current_x, current_y))
        for dx, dy in individual:
            current_x += dx
            current_y += dy
            positions.append((current_x, current_y))

        remaining_data = user_tasks.copy()
        completion_time = num_time_steps
        transmission_rates = []

        for t in range(num_time_steps):
            pos = positions[t]
            for i in range(len(user_positions)):
                if remaining_data[i] <= 0.001 * len(user_positions):
                    transmission_rates.append(0)
                    continue
                user_pos = user_positions[i]
                distance = math.sqrt((pos[0] - user_pos[0]) ** 2 + (pos[1] - user_pos[1]) ** 2)
                if distance < 0.001:
                    distance = 0.001

                h = h0 / (distance ** 2) * _fading_gain(shadow_std, rician_k)
                rate = math.log(1 + h / sigma)
                data_transmitted = rate * time_step_duration

                if data_transmitted > remaining_data[i]:
                    data_transmitted = remaining_data[i]
                remaining_data[i] -= data_transmitted
                transmission_rates.append(rate)

            if all(d <= 0.001 * len(user_positions) for d in remaining_data):
                completion_time = t + 1
                break

        return completion_time, transmission_rates

    return analyze_solution_stochastic


def patch_ga(shadow_std=DEFAULT_SHADOW_STD, rician_k=DEFAULT_RICIAN_K):
    _ga.eval_fitness_improved = _make_eval_fitness_stochastic(shadow_std, rician_k)
    _ga.analyze_solution = _make_analyze_solution_stochastic(shadow_std, rician_k)


patch_ga()

ga_optimize_extended = _ga.ga_optimize

"""Smoke test the stochastic-channel-patched DQN/A2C/DDPG baseline adapters."""
import math

from dqn_optimization_extended import dqn_optimize_extended
from a2c_optimization_extended import a2c_optimize_extended
from ddpg_optimization_extended import ddpg_optimize_extended

print("=== SMOKE: DQN extended ===")
r = dqn_optimize_extended(num_users=4, area_size=10, h_unit=1, sigma=0.1,
                           max_step=0.2, time_step_duration=0.1, max_episodes=5)
print(r["time_steps"], r["avg_task_reduction"])
assert math.isfinite(r["avg_task_reduction"])

print("=== SMOKE: A2C extended ===")
r = a2c_optimize_extended(num_users=4, area_size=10, h_unit=1, sigma=0.1,
                           max_step=0.2, time_step_duration=0.1, max_episodes=5)
print(r["time_steps"], r["avg_task_reduction"])
assert math.isfinite(r["avg_task_reduction"])

print("=== SMOKE: DDPG extended ===")
r = ddpg_optimize_extended(num_users=4, area_size=10, h_unit=1, sigma=0.1,
                            max_step=0.2, time_step_duration=0.1, max_episodes=5)
print(r["time_steps"], r["avg_task_reduction"])
assert math.isfinite(r["avg_task_reduction"])

print("Smoke test PASSED")

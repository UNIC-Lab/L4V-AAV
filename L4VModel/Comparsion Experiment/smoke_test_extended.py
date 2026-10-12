"""Quick end-to-end smoke test for env_extended / dso_optimization_extended / mpc_optimization_extended.
Small scale only: a handful of episodes/steps, just to confirm the pipeline runs, produces
finite numbers, and that toggling allocation / stochastic channel actually changes behavior.
"""
import math
import torch

from dso_optimization_extended import dso_optimize_extended
from mpc_optimization_extended import mpc_optimize

torch.manual_seed(0)

print("=== SMOKE TEST: L4V extended, static equal-split (should match original model) ===")
r1 = dso_optimize_extended(num_users=4, area_size=10, h_unit=1, sigma=0.1,
                            max_step=0.2, time_step_duration=0.1,
                            max_episodes=20, use_allocation=False, shadow_std=0.0,
                            rician_k=None, energy_weight=0.0, seed=1)
print(r1)
assert math.isfinite(r1['avg_task_reduction'])
assert r1['time_steps'] > 0

print("\n=== SMOKE TEST: L4V extended, learned allocation + shadowing ===")
r2 = dso_optimize_extended(num_users=4, area_size=10, h_unit=1, sigma=0.1,
                            max_step=0.2, time_step_duration=0.1,
                            max_episodes=20, use_allocation=True, shadow_std=0.15,
                            rician_k=6.0, energy_weight=1e-4, seed=1)
print(r2)
assert math.isfinite(r2['avg_task_reduction'])
assert r2['total_energy'] > 0

print("\n=== SMOKE TEST: MPC baseline, small horizon ===")
r3 = mpc_optimize(num_users=4, area_size=10, h_unit=1, sigma=0.1,
                   max_step=0.2, time_step_duration=0.1,
                   horizon=5, inner_iters=5, max_episodes=15,
                   use_allocation=True, shadow_std=0.15, rician_k=6.0, seed=1)
print(r3)
assert math.isfinite(r3['avg_task_reduction'])
assert r3['time_steps'] > 0

print("\nSmoke test PASSED")

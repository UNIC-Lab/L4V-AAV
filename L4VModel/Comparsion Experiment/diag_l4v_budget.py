"""Stage-1 cheap validation: does a larger training budget actually fix L4V's
non-convergence at sigma=0.1/0.25 seen at max_episodes=250, before committing
compute to a full sweep at that budget."""
import time
from dso_optimization_extended import dso_optimize_extended

for sigma in [0.1, 0.25]:
    for budget in [400, 600]:
        t0 = time.time()
        r = dso_optimize_extended(num_users=4, area_size=10, h_unit=1, sigma=sigma,
                                   max_step=0.2, time_step_duration=0.1, max_episodes=budget,
                                   use_allocation=True, shadow_std=0.15, rician_k=6.0,
                                   energy_weight=0.0, seed=7)
        print(f"sigma={sigma} budget={budget} time_steps={r['time_steps']} "
              f"wall={time.time()-t0:.1f}s")

import math
import time
from ga_optimization_extended import ga_optimize_extended

t0 = time.time()
r = ga_optimize_extended(num_users=4, area_size=10, h_unit=1, sigma=0.1,
                          max_step=0.2, time_step_duration=0.1, max_generations=5)
print("time_steps", r["time_steps"], "avg_task_reduction", r["avg_task_reduction"],
      "wall", time.time() - t0)
assert math.isfinite(r["avg_task_reduction"])
print("Smoke test PASSED")

"""
orchestrate_reruns.py

Runs the three heavy full-budget reruns strictly sequentially in one process
(so at most one heavy 2000-episode training job is on the GPU at a time, while
the lightweight MPC deterministic sweeps run in their own processes):

    1. run_alloc_full   (paired-instance allocation ablation, K in {4,8})
    2. run_energy_full  (paired-instance completion-time-vs-energy sweep)
    3. run_trajectory_case (executed paths on one shared instance)

Each stage saves its own CSV/NPZ as it finishes, so partial progress is never
lost if the process is interrupted.
"""
import time

import run_alloc_full
import run_energy_full


def main():
    t0 = time.time()
    print(f"[orchestrator] START", flush=True)

    print(f"[orchestrator] stage 1/3: allocation ablation", flush=True)
    run_alloc_full.run(num_users_list=[4, 8], n_repeat=5, max_episodes=2000)
    print(f"[orchestrator] stage 1 done at {time.time()-t0:.0f}s", flush=True)

    print(f"[orchestrator] stage 2/3: energy sweep", flush=True)
    run_energy_full.run(gamma_list=[0.0, 1e-5, 1e-4, 3e-4, 1e-3], n_repeat=5,
                         max_episodes=2000, v_scale=1.0)
    print(f"[orchestrator] stage 2 done at {time.time()-t0:.0f}s", flush=True)

    print(f"[orchestrator] stage 3/3: trajectory case", flush=True)
    import run_trajectory_case
    run_trajectory_case.main(smoke=False)
    print(f"[orchestrator] ALL DONE at {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

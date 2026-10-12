"""
run_latency.py

Separates the two cost metrics the compute table previously mixed:
  - Training latency: one-off wall-clock to train the policy (L4V, DQN, A2C,
    DDPG). GA and MPC have no reusable policy, so this is not applicable.
  - Inference latency: per-mission deployment cost. For the neural policies this
    is a forward-pass rollout over the horizon (measured here, timed with CUDA
    synchronization, warm-started); for GA and MPC it is the full per-instance
    solve, since they re-optimize from scratch on every mission.

Training-latency numbers are read from the existing default-config sweep CSVs
(num_users=4 == the default K=4, L=10, eta=1, sigma^2=0.1 operating point);
inference latency for the neural policies is measured here. Prints a LaTeX-ready
table body.
"""
import os
import time
import math
import numpy as np
import pandas as pd
import torch

from dso_optimization import PathPlanningNet, compute_transmission_rate, g

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "results/data"
NUM_USERS, AREA, H_UNIT, SIGMA, V_MAX, DT, HORIZON = 4, 10.0, 1.0, 0.1, 0.2, 0.1, 500
N_REPEAT = 30


def med_train(tag):
    p = os.path.join(DATA, f"{tag}_result_num_users_4.csv")
    return pd.read_csv(p).total_training_time.median()


def fixed_instance(seed=777):
    torch.manual_seed(seed)
    up = (torch.rand((NUM_USERS, 2), device=device) - 0.5) * AREA
    tasks = torch.rand(NUM_USERS, device=device) * 2 + 2
    return up, tasks


def time_l4v_inference(up, tasks):
    """Median wall-clock of one full forward-pass rollout of the trained-shape policy."""
    model = PathPlanningNet(NUM_USERS, v_max=V_MAX).to(device).eval()
    times = []
    for r in range(N_REPEAT + 5):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            base = torch.zeros((1, 2), device=device); rem = tasks.clone()
            for t in range(HORIZON):
                angle, speed, _ = model(base, up, rem)
                base = g(base, angle, speed)
                rem = torch.clamp(rem - compute_transmission_rate(base, up, H_UNIT, SIGMA) * DT, min=0)
                if torch.sum(rem) < 1e-3 * NUM_USERS:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize()
        if r >= 5:  # discard warm-up
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_agent_inference(make_agent, is_a2c, up, tasks):
    agent = make_agent()
    times = []
    for r in range(N_REPEAT + 5):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        base = torch.zeros((2,), device=device); rem = tasks.clone()
        state = agent.get_state(base, up, rem)
        for t in range(HORIZON):
            s_in = state.cpu().numpy() if (torch.is_tensor(state) and state.is_cuda) else state
            sel = agent.select_action(s_in, eval_mode=True)
            a = sel[0] if is_a2c else sel
            nb, nr, rw, ns, done = agent.step(base, a, up, rem)
            state = ns; base = nb; rem = nr
            if done:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        if r >= 5:
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_ddpg_inference(make_agent, up, tasks):
    agent = make_agent()
    times = []
    for r in range(N_REPEAT + 5):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        base = torch.zeros((2,), device=device); rem = tasks.clone()
        state = agent.get_state(base, up, rem)
        for t in range(HORIZON):
            s_in = state.cpu().numpy() if (torch.is_tensor(state) and state.is_cuda) else state
            a = agent.select_action(s_in, add_noise=False)
            nb, nr, rw, ns, done = agent.step(base, a, up, rem)
            state = ns; base = nb; rem = nr
            if done:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        if r >= 5:
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


if __name__ == "__main__":
    up, tasks = fixed_instance()
    ht = torch.tensor(H_UNIT, device=device); st = torch.tensor(SIGMA, device=device)

    from dqn_optimization import DQNAgent
    from a2c_optimization import A2CAgent
    from ddpg_optimization import DDPGAgent
    sdim = 2 + NUM_USERS * 2 + NUM_USERS

    inf = {}
    inf["L4V"] = time_l4v_inference(up, tasks)
    inf["DQN"] = time_agent_inference(
        lambda: DQNAgent(sdim, 256, NUM_USERS, AREA, ht, st, V_MAX, DT), False, up, tasks)
    inf["A2C"] = time_agent_inference(
        lambda: A2CAgent(sdim, 8, NUM_USERS, AREA, ht, st, V_MAX, DT), True, up, tasks)
    inf["DDPG"] = time_ddpg_inference(
        lambda: DDPGAgent(sdim, 2, NUM_USERS, AREA, ht, st, V_MAX, DT), up, tasks)

    train = {m: med_train(t) for m, t in [("L4V", "dso"), ("DQN", "dqn"), ("A2C", "a2c"), ("DDPG", "ddpg")]}
    ga_solve = med_train("ga")   # GA re-solves per instance -> this IS its inference cost
    mpc_solve = med_train("mpc")

    # The untrained policy runs the full HORIZON; a trained one terminates at its median
    # completion length, so scale the measured rollout time to a realistic per-mission cost.
    def med_steps(tag):
        return pd.read_csv(os.path.join(DATA, f"{tag}_result_num_users_4.csv")).time_steps.median()
    steps = {"L4V": med_steps("dso"), "DQN": med_steps("dqn"),
             "A2C": med_steps("a2c"), "DDPG": med_steps("ddpg")}
    inf = {m: inf[m] * steps[m] / HORIZON for m in inf}  # per-mission at median length

    print("=== training latency (s), default config ===")
    for m in ["L4V", "DQN", "A2C", "DDPG"]:
        print(f"  {m}: {train[m]:.0f}")
    print("=== inference latency (ms/mission) ===")
    for m in ["L4V", "DQN", "A2C", "DDPG"]:
        print(f"  {m}: {inf[m]*1000:.1f} ms")
    print(f"  GA per-mission solve: {ga_solve:.0f} s")
    print(f"  MPC per-mission solve: {mpc_solve:.0f} s")

    # LaTeX body: Training (s) | Inference (per mission)
    def infcell(ms):
        return f"{ms*1000:.0f}~ms"
    rows = [
        ("L4V", f"{train['L4V']:.0f}", infcell(inf['L4V'])),
        ("GA", "--", f"{ga_solve:.0f}~s"),
        ("DQN", f"{train['DQN']:.0f}", infcell(inf['DQN'])),
        ("A2C", f"{train['A2C']:.0f}", infcell(inf['A2C'])),
        ("DDPG", f"{train['DDPG']:.0f}", infcell(inf['DDPG'])),
        ("MPC", "--", f"{mpc_solve:.0f}~s"),
    ]
    print("\n=== LaTeX table body ===")
    for m, tr, it in rows:
        print(f"{m} & {tr} & {it} \\\\")

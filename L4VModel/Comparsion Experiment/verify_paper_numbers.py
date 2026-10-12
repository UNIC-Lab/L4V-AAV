"""
verify_paper_numbers.py

Independently recomputes every quantitative claim in ref.tex from the source
CSVs and flags mismatches. Read-only; prints PASS/FAIL per checked number.
"""
import os
import pandas as pd
import numpy as np

DATA = "results/data"
EXT = "results_extended/data"
fails = []


def med_time(tag, param, val):
    for c in [f"{tag}_result_{param}_{val}.csv", f"{tag}_result_{param}_{val:g}.csv"]:
        p = os.path.join(DATA, c)
        if os.path.exists(p):
            return pd.read_csv(p).time_steps.median()
    return None


def check(name, got, expect, tol=1.0):
    ok = got is not None and abs(got - expect) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: computed={got}, paper={expect}")
    if not ok:
        fails.append(name)


# --- Sweep gain claims (L4V vs best learned baseline GA/DQN/A2C/DDPG) ---
def gain(param, vals):
    out = []
    for v in vals:
        l4v = med_time("dso", param, v)
        base = [med_time(b, param, v) for b in ["ga", "dqn", "a2c", "ddpg"]]
        base = [b for b in base if b is not None]
        out.append(round((min(base) - l4v) / min(base) * 100))
    return out


print("=== Sweep gains (percent, endpoints cited in prose) ===")
g_area = gain("area_size", [5, 10, 15, 20, 25])
check("area gain L=5", g_area[0], 10); check("area gain L=25", g_area[4], 65)
g_sig = gain("sigma", [0.05, 0.1, 0.15, 0.2, 0.25])
check("sigma gain 0.05", g_sig[0], 14); check("sigma gain 0.2", g_sig[3], 41)
g_usr = gain("num_users", [2, 4, 6, 8, 10])
check("users gain min", min(g_usr), 35); check("users gain max", max(g_usr), 46)
g_h = gain("h_unit", [0.5, 1.0, 1.5, 2.0, 2.5])
check("hunit gain 0.5", g_h[0], 64); check("hunit gain 2.5", g_h[4], 13)

# --- Compute table (medians, seconds) ---
print("\n=== Table I compute cost (median s) ===")
def med_train(tag, param, val):
    for c in [f"{tag}_result_{param}_{val}.csv", f"{tag}_result_{param}_{val:g}.csv"]:
        p = os.path.join(DATA, c)
        if os.path.exists(p):
            return pd.read_csv(p).total_training_time.median()
    return None
tbl = {"L4V": "dso", "GA": "ga", "DQN": "dqn", "A2C": "a2c", "DDPG": "ddpg", "MPC": "mpc"}
paper_tbl = {  # (sig0.05, sig0.25, K2, K10)
    "L4V": (144, 467, 79, 294), "GA": (1601, 1467, 1590, 3059),
    "DQN": (1141, 2002, 1018, 88), "A2C": (150, 8440, 493, 1954),
    "DDPG": (4577, 11007, 2206, 6714), "MPC": (61, 172, 59, 290)}
for lab, tag in tbl.items():
    got = (med_train(tag, "sigma", 0.05), med_train(tag, "sigma", 0.25),
           med_train(tag, "num_users", 2), med_train(tag, "num_users", 10))
    for gi, pe, col in zip(got, paper_tbl[lab], ["s05", "s25", "K2", "K10"]):
        check(f"compute {lab} {col}", gi, pe, tol=2.0)

# --- Allocation (full-budget paired medians) ---
print("\n=== Allocation ablation (median steps/energy) ===")
for K, t_off, t_on, e_off, e_on in [(4, 51, 40, 859, 674), (8, 73, 52, 1230, 876)]:
    on = pd.read_csv(os.path.join(EXT, f"allocfull_users{K}_allocTrue.csv"))
    off = pd.read_csv(os.path.join(EXT, f"allocfull_users{K}_allocFalse.csv"))
    check(f"alloc K={K} time on", on.time_steps.median(), t_on)
    check(f"alloc K={K} time off", off.time_steps.median(), t_off)
    check(f"alloc K={K} energy on", on.total_energy.median(), e_on, tol=3)
    check(f"alloc K={K} energy off", off.total_energy.median(), e_off, tol=3)

# --- Energy sweep (full-budget paired medians) ---
print("\n=== Energy tradeoff (median steps/energy) ===")
for g, t, e in [(0.0, 40, 674), (1e-4, 36, 606), (1e-3, 37, 623)]:
    df = pd.read_csv(os.path.join(EXT, f"energyfull_gamma{g}.csv"))
    conv = df[df.time_steps < 500]
    check(f"energy g={g:g} time", conv.time_steps.median(), t)
    check(f"energy g={g:g} energy", conv.total_energy.median(), e, tol=3)

# --- Trajectory step counts ---
print("\n=== Trajectory case (steps) ===")
d = np.load(os.path.join(EXT, "trajectory_case.npz"), allow_pickle=True)
for m, s in [("L4V", 51), ("MPC", 56), ("DDPG", 230)]:
    check(f"traj {m} steps", len(d[f"path_{m}"]) - 1, s)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))

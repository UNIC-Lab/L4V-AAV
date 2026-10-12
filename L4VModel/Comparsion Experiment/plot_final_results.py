"""
plot_final_results.py

Single source of truth for every data figure and results table in the revised
paper. Produces, from real experiment CSVs:

  Figs 2-5 : four parameter sweeps (area size, noise power, user density,
             channel gain), each a 3-panel figure (average data collection
             rate, completion time, compute time) for all six methods
             L4V/GA/DQN/A2C/DDPG/MPC
             as median lines with inter-quartile shaded bands. Completion-time
             and compute-time panels use a log y-axis so the multiplicative
             gaps read as clean monotone trends rather than a compressed
             tangle near the axis.
  Fig alloc: full-budget paired-instance allocation ablation with compact
             broken axes, explicit failure cases, medians, and semantic legend.
  Fig energy: full-budget, paired-instance completion-time-vs-energy sweep.
  Fig traj : executed AAV paths of L4V/MPC/DDPG on one shared instance.
  Tables   : per-sweep completion-time table with an L4V gain row.

Metric note: the "average data collection rate" is total delivered data
divided by mission completion time (the CSV column avg_task_reduction). It is
the effective end-to-end delivery rate of the whole mission and, unlike a
per-position instantaneous rate, is not confounded by how many steps the
mission happens to take, so it is directly comparable across methods.

All figures are written as BOTH .pdf (for the paper) and .png (for review) to
paper/figs/<group>/, reusing the shared palette in style.py. No titles, font
sizes >= 12 pt, per the project visualization standard.
"""
import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, LogLocator

from style import METHOD_COLORS, METHOD_ORDER

DATA_DIR = "results/data"                     # original full-budget n=10 sweeps + MPC
EXT_DIR = "results_extended/data"             # full-budget alloc / energy / trajectory
FIG_ROOT = os.path.join("..", "..", "..", "..", "paper", "figs")
TABLE_ROOT = "paper_assets/tables"

# CSV method tag -> paper label
TAG2LABEL = {"dso": "L4V", "ga": "GA", "dqn": "DQN", "a2c": "A2C", "ddpg": "DDPG", "mpc": "MPC"}
LABEL_ORDER = ["L4V", "GA", "DQN", "A2C", "DDPG", "MPC"]

SWEEPS = {
    "area_size": ([5, 10, 15, 20, 25], "Service-area size $L$"),
    "sigma":     ([0.05, 0.1, 0.15, 0.2, 0.25], r"Noise power $\sigma^2$"),
    "num_users": ([2, 4, 6, 8, 10], "Number of users $K$"),
    "h_unit":    ([0.5, 1.0, 1.5, 2.0, 2.5], r"Unit-distance channel gain $\eta$"),
}

MARKERS = {"L4V": "o", "GA": "s", "DQN": "^", "A2C": "D", "DDPG": "v", "MPC": "P"}


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, out_dir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {os.path.join(out_dir, stem)}.{{pdf,png}}")


def load(tag, param, value):
    """Robust load of one (method, param, value) CSV across filename variants."""
    cands = [f"{tag}_result_{param}_{value}.csv",
             f"{tag}_result_{param}_{value:g}.csv"]
    if float(value) == int(value):
        cands.append(f"{tag}_result_{param}_{int(value)}.csv")
        cands.append(f"{tag}_result_{param}_{int(value)}.0.csv")
    for c in cands:
        p = os.path.join(DATA_DIR, c)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


# ---------------------------------------------------------------------------
# Figs 2-5 : parameter sweeps, 6-method median + IQR
# ---------------------------------------------------------------------------
def plot_sweep(param, stem_suffix=""):
    values, xlabel = SWEEPS[param]
    out_dir = _ensure(os.path.join(FIG_ROOT, param.replace("_size", "").replace("num_users", "num_user")))
    # map to the figs/ subdir names the paper uses
    out_dir = _ensure(os.path.join(FIG_ROOT, {"area_size": "area", "sigma": "sigma",
                                              "num_users": "num_user", "h_unit": "h_unit"}[param]))

    # metric column, label, log-scale, prettified file stem
    panels = [
        ("avg_task_reduction", "Avg. Data Collection Rate", False, "rate"),
        ("time_steps", "Completion Time (steps)", True, "time_step"),
        ("total_training_time", "Compute Time (s)", True, "train_time"),
    ]
    present = {}
    for tag in TAG2LABEL:
        lab = TAG2LABEL[tag]
        ok = False
        med_by_metric = {}
        for metric, _, _, _ in panels:
            med, q1, q3 = [], [], []
            for v in values:
                df = load(tag, param, v)
                if df is None or metric not in df:
                    med.append(np.nan); q1.append(np.nan); q3.append(np.nan); continue
                vals = df[metric].dropna().values
                ok = True
                med.append(np.median(vals)); q1.append(np.percentile(vals, 25)); q3.append(np.percentile(vals, 75))
            med_by_metric[metric] = (np.array(med), np.array(q1), np.array(q3))
        if ok:
            present[lab] = med_by_metric

    stem_map = {"area_size": "area_size", "sigma": "sigma", "num_users": "num_users", "h_unit": "h_unit"}
    for metric, ylabel, logy, panel_stem in panels:
        fig, ax = plt.subplots(figsize=(6, 4.8))
        for lab in LABEL_ORDER:
            if lab not in present:
                continue
            med, q1, q3 = present[lab][metric]
            valid = ~np.isnan(med)
            xv = np.array(values)[valid]
            ax.plot(xv, med[valid], marker=MARKERS[lab], markersize=7, linewidth=2,
                    label=lab, color=METHOD_COLORS[lab])
            ax.fill_between(xv, q1[valid], q3[valid], alpha=0.15, color=METHOD_COLORS[lab])
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if logy:
            ax.set_yscale("log")
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=11, ncol=2)
        ax.text(0.02, 0.03, "$n=10$; median [IQR]", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.75", alpha=0.9))
        if metric == "time_steps" and "L4V" in present:
            ours = present["L4V"][metric][0]
            competitors = np.vstack([
                present[lab][metric][0] for lab in LABEL_ORDER
                if lab != "L4V" and lab in present
            ])
            best = np.nanmin(competitors, axis=0)
            gain = 100.0 * (best - ours) / best
            ax.text(0.98, 0.04,
                    f"L4V gain vs. best comparator:\n"
                    f"{gain[0]:.0f}% $\\rightarrow$ {gain[-1]:.0f}%",
                    transform=ax.transAxes, ha="right", va="bottom",
                    color=METHOD_COLORS["L4V"], fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=METHOD_COLORS["L4V"], alpha=0.9))
        plt.tight_layout()
        _save(fig, out_dir, f"{stem_map[param]}_{panel_stem}{stem_suffix}")


def table_sweep(param):
    values, _ = SWEEPS[param]
    rows = []
    med = {}
    for tag, lab in TAG2LABEL.items():
        med[lab] = []
        for v in values:
            df = load(tag, param, v)
            med[lab].append(np.median(df["time_steps"].dropna().values) if df is not None else np.nan)
    for lab in LABEL_ORDER:
        if all(np.isnan(med.get(lab, [np.nan]))):
            continue
        rows.append([lab] + [f"{x:.0f}" if not np.isnan(x) else "--" for x in med[lab]])
    # gain row: L4V vs best-of-baselines (exclude MPC and L4V)
    gain = ["L4V gain over best learned baseline"]
    for j, v in enumerate(values):
        l4v = med["L4V"][j]
        base = [med[b][j] for b in ["GA", "DQN", "A2C", "DDPG"] if b in med and not np.isnan(med[b][j])]
        if base and not np.isnan(l4v) and min(base) > 0:
            gain.append(f"{(min(base) - l4v) / min(base) * 100:+.0f}%")
        else:
            gain.append("--")
    rows.append(gain)
    os.makedirs(TABLE_ROOT, exist_ok=True)
    hdr = ["Method"] + [f"{param}={v:g}" for v in values]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    with open(os.path.join(TABLE_ROOT, f"table_{param}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved table_{param}.md")


# ---------------------------------------------------------------------------
# Fig alloc : full-budget paired allocation ablation
# ---------------------------------------------------------------------------
def plot_alloc():
    files = glob.glob(os.path.join(EXT_DIR, "allocfull_users*_alloc*.csv"))
    files = [f for f in files if "_smoke" not in f]
    if not files:
        print("[alloc] no full-budget data yet, skipping")
        return
    out_dir = _ensure(os.path.join(FIG_ROOT, "alloc"))
    users = sorted({int(os.path.basename(f).split("users")[1].split("_")[0]) for f in files})
    C_ON, C_OFF, C_REGRESS = "#D62728", "#4C78A8", "#4D4D4D"
    table = []
    for metric, ylabel, stem in [("avg_task_reduction", "Avg. Data Collection Rate", "alloc_rate"),
                                 ("time_steps", "Completion Time (steps)", "alloc_time_step"),
                                 ("total_energy", "Total Flight Energy", "alloc_energy")]:
        broken_axis = metric in {"time_steps", "total_energy"}
        if broken_axis:
            fig = plt.figure(figsize=(6.4, 4.5))
            grid = fig.add_gridspec(
                2, 1, height_ratios=[0.95, 3.05], hspace=0.035
            )
            ax_top = fig.add_subplot(grid[0])
            ax = fig.add_subplot(grid[1], sharex=ax_top)
            plot_axes = [ax_top, ax]
        else:
            fig, ax = plt.subplots(figsize=(6.4, 4.5))
            ax_top = None
            plot_axes = [ax]
        xticks, xlabels = [], []
        for j, K in enumerate(users):
            x_equal, x_learned = 3 * j, 3 * j + 1
            on = pd.read_csv(os.path.join(EXT_DIR, f"allocfull_users{K}_allocTrue.csv"))
            off = pd.read_csv(os.path.join(EXT_DIR, f"allocfull_users{K}_allocFalse.csv"))
            learned = on[metric].to_numpy(dtype=float)
            equal = off[metric].to_numpy(dtype=float)
            n = min(len(learned), len(equal))
            learned, equal = learned[:n], equal[:n]
            for trial, (y0, y1) in enumerate(zip(equal, learned), start=1):
                improved = y1 < y0 if metric != "avg_task_reduction" else y1 > y0
                line_color = C_ON if improved else C_REGRESS
                for target_ax in plot_axes:
                    target_ax.plot(
                        [x_equal, x_learned], [y0, y1],
                        color=line_color, alpha=0.42, linewidth=1.25, zorder=1
                    )
                    target_ax.scatter(
                        [x_equal], [y0], s=34, facecolor="white",
                        edgecolor=C_OFF, linewidth=1.2, zorder=3
                    )
                    target_ax.scatter(
                        [x_learned], [y1], s=38, color=C_ON,
                        edgecolor="white", linewidth=0.5, zorder=3
                    )
                if not improved and metric in {"time_steps", "total_energy"}:
                    failure_label = (
                        "500-slot cap" if metric == "time_steps" and y1 >= 499
                        else "capped run" if metric == "total_energy" and y1 >= 8000
                        else "regression"
                    )
                    failure_ax = ax_top if "cap" in failure_label else ax
                    failure_ax.annotate(
                        failure_label,
                        (x_learned, y1),
                        xytext=(10, -22 if "cap" in failure_label else 7),
                        textcoords="offset points",
                        fontsize=8.0,
                        color=C_REGRESS,
                        arrowprops=dict(arrowstyle="-", color=C_REGRESS, lw=0.7),
                    )
            med_equal, med_learned = np.median(equal), np.median(learned)
            ax.plot([x_equal - 0.22, x_equal + 0.22], [med_equal, med_equal],
                    color=C_OFF, linewidth=4, solid_capstyle="round", zorder=4)
            ax.plot([x_learned - 0.22, x_learned + 0.22], [med_learned, med_learned],
                    color=C_ON, linewidth=4, solid_capstyle="round", zorder=4)
            better = np.sum(learned < equal) if metric != "avg_task_reduction" else np.sum(learned > equal)
            if metric == "avg_task_reduction":
                change = 100.0 * (med_learned - med_equal) / med_equal
            else:
                change = 100.0 * (med_equal - med_learned) / med_equal
            annotation_ax = ax_top if broken_axis else ax
            annotation_ax.text(
                0.28 + 0.46 * j, 0.88,
                f"median gain {change:.0f}%\n{better}/{n} trials improve",
                transform=annotation_ax.transAxes, ha="center", va="top",
                fontsize=8.0, color=C_ON if better >= n / 2 else C_REGRESS,
                zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                          pad=0.25),
            )
            xticks.extend([x_equal, x_learned])
            xlabels.extend([f"Equal\n$K={K}$", f"Learned\n$K={K}$"])
            if metric == "time_steps":
                table.extend([
                    [K, "Learned alloc.", f"{on[metric].mean():.1f}", f"{on[metric].std():.1f}",
                     f"{on['total_energy'].mean():.0f}", f"{on['avg_task_reduction'].mean():.3f}"],
                    [K, "Equal split", f"{off[metric].mean():.1f}", f"{off[metric].std():.1f}",
                     f"{off['total_energy'].mean():.0f}", f"{off['avg_task_reduction'].mean():.3f}"],
                ])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=10)
        if broken_axis:
            fig.supylabel(ylabel, x=0.025, fontsize=12)
            if metric == "time_steps":
                ax.set_ylim(20, 150)
                ax_top.set_ylim(460, 525)
                ax.set_yticks([20, 40, 60, 80, 100, 120, 140])
                ax_top.set_yticks([500])
            else:
                ax.set_ylim(380, 2500)
                ax_top.set_ylim(7800, 9200)
                ax.set_yticks([400, 700, 1000, 1500, 2000, 2500])
                ax_top.set_yticks([8000, 9000])
            for target_ax in plot_axes:
                target_ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{int(value):,}")
                )
                target_ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax_top.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            # Paired diagonal cuts signal the omitted empty y-range without
            # hiding the capped observation.
            cut = 0.010
            cut_kw = dict(color="0.2", clip_on=False, linewidth=0.8)
            ax_top.plot((-cut, +cut), (-cut, +cut),
                        transform=ax_top.transAxes, **cut_kw)
            ax_top.plot((1 - cut, 1 + cut), (-cut, +cut),
                        transform=ax_top.transAxes, **cut_kw)
            ax.plot((-cut, +cut), (1 - cut, 1 + cut),
                    transform=ax.transAxes, **cut_kw)
            ax.plot((1 - cut, 1 + cut), (1 - cut, 1 + cut),
                    transform=ax.transAxes, **cut_kw)
        else:
            ax.set_ylabel(ylabel, fontsize=12)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        for target_ax in plot_axes:
            target_ax.tick_params(labelsize=9.5)
            target_ax.tick_params(
                which="both", direction="in", top=True, right=True
            )
            target_ax.tick_params(which="minor", length=2.5)
            target_ax.tick_params(which="major", length=4)
            target_ax.axvline(2.0, color="0.86", linewidth=0.9, zorder=0)
            target_ax.grid(
                True, axis="y", alpha=0.24, which="major", linewidth=0.8
            )
            target_ax.grid(
                True, axis="y", alpha=0.10, which="minor", linewidth=0.5
            )
        if broken_axis:
            legend_handles = [
                Line2D([], [], marker="o", linestyle="none", markersize=5.2,
                       markerfacecolor="white", markeredgecolor=C_OFF,
                       label="Equal split"),
                Line2D([], [], marker="o", linestyle="none", markersize=5.2,
                       markerfacecolor=C_ON, markeredgecolor="white",
                       label="Learned allocation"),
                Line2D([], [], color=C_ON, alpha=0.55, linewidth=1.2,
                       label="Paired trial"),
                Line2D([], [], color="0.3", linewidth=3.2,
                       solid_capstyle="round", label="Median"),
            ]
            ax.legend(
                handles=legend_handles, loc="upper left",
                bbox_to_anchor=(0.01, 0.91), ncol=2,
                frameon=False, fontsize=9.2, handlelength=1.8,
                columnspacing=0.9, labelspacing=0.25, borderaxespad=0.35
            )
        panel_label = {"time_steps": "(a)", "total_energy": "(b)",
                       "avg_task_reduction": "(a)"}[metric]
        label_ax = ax_top if broken_axis else ax
        label_ax.text(
            0.012, 0.82 if broken_axis else 0.975,
            panel_label, transform=label_ax.transAxes,
            ha="left", va="top", fontsize=10.5, fontweight="bold"
        )
        if broken_axis:
            fig.subplots_adjust(left=0.15, right=0.985, top=0.985, bottom=0.16)
        else:
            plt.tight_layout()
        _save(fig, out_dir, stem)
    if table:
        os.makedirs(TABLE_ROOT, exist_ok=True)
        with open(os.path.join(TABLE_ROOT, "table_alloc.md"), "w", encoding="utf-8") as f:
            f.write("| K | Policy | Steps mean | Steps std | Energy mean | Rate mean |\n|---|---|---|---|---|---|\n")
            for r in table:
                f.write("| " + " | ".join(str(c) for c in r) + " |\n")
        print("saved table_alloc.md")


# ---------------------------------------------------------------------------
# Fig energy : full-budget paired energy sweep
# ---------------------------------------------------------------------------
def plot_energy():
    files = glob.glob(os.path.join(EXT_DIR, "energyfull_gamma*.csv"))
    files = [f for f in files if "_smoke" not in f]
    if not files:
        print("[energy] no full-budget data yet, skipping")
        return
    out_dir = _ensure(os.path.join(FIG_ROOT, "energy"))
    gammas = sorted({float(os.path.basename(f).split("gamma")[1].replace(".csv", "")) for f in files})
    frames = []
    for g in gammas:
        df = pd.read_csv(os.path.join(EXT_DIR, f"energyfull_gamma{g}.csv"))
        df = df.copy()
        df["gamma"] = g
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    xpos = np.arange(len(gammas), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    baseline = all_df[all_df["gamma"] == gammas[0]].sort_values("run_index")
    metric_styles = {
        "time_steps": ("Completion time", METHOD_COLORS["L4V"], "o", -0.045, "-"),
        "total_energy": ("Flight energy", METHOD_COLORS["MPC"], "s", 0.045, "--"),
    }
    for metric, (label, color, marker, xoff, linestyle) in metric_styles.items():
        trial_changes = []
        for run_idx in sorted(baseline["run_index"].unique()):
            base_val = float(baseline.loc[baseline["run_index"] == run_idx, metric].iloc[0])
            vals = []
            for g in gammas:
                row = all_df[(all_df["gamma"] == g) & (all_df["run_index"] == run_idx)]
                vals.append(100.0 * (float(row[metric].iloc[0]) - base_val) / base_val)
            vals = np.asarray(vals)
            trial_changes.append(vals)
            ax.plot(xpos + xoff, vals, color=color, alpha=0.16,
                    linewidth=1, linestyle=linestyle)
        trial_changes = np.vstack(trial_changes)
        med = np.median(trial_changes, axis=0)
        q1 = np.percentile(trial_changes, 25, axis=0)
        q3 = np.percentile(trial_changes, 75, axis=0)
        ax.fill_between(xpos + xoff, q1, q3, color=color, alpha=0.14)
        ax.plot(xpos + xoff, med, color=color, marker=marker,
                linestyle=linestyle, linewidth=2.4, markersize=7,
                label=f"{label}: median [IQR]")
    ax.axhline(0, color="0.25", linestyle="--", linewidth=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{g:g}" for g in gammas], rotation=20, ha="right")
    ax.set_xlabel(r"Energy weight $\gamma$", fontsize=14)
    ax.set_ylabel(r"Change from $\gamma=0$ (%)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    _save(fig, out_dir, "energy_time_pareto")


# ---------------------------------------------------------------------------
# Fig traj : executed paths on one shared instance
# ---------------------------------------------------------------------------
def plot_trajectory():
    p = os.path.join(EXT_DIR, "trajectory_case.npz")
    if not os.path.exists(p):
        print("[traj] no data yet, skipping")
        return
    out_dir = _ensure(os.path.join(FIG_ROOT, "traj"))
    d = np.load(p, allow_pickle=True)
    up = d["user_positions"]; tasks = d["user_tasks"]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for lab in LABEL_ORDER:  # L4V, GA, DQN, A2C, DDPG, MPC
        key = f"path_{lab}"
        if key not in d:
            continue
        path = d[key]
        ax.plot(path[:, 0], path[:, 1], "-", linewidth=2, alpha=0.85,
                label=f"{lab} ({len(path)-1} steps)", color=METHOD_COLORS[lab])
        ax.scatter(path[-1, 0], path[-1, 1], marker=MARKERS[lab], s=90,
                   color=METHOD_COLORS[lab], edgecolor="k", zorder=5)
        if lab in {"L4V", "MPC"} and len(path) > 4:
            for frac in (0.35, 0.70):
                idx = min(len(path) - 2, max(0, int(frac * (len(path) - 1))))
                ax.annotate("", xy=path[idx + 1], xytext=path[idx],
                            arrowprops=dict(arrowstyle="-|>", color=METHOD_COLORS[lab],
                                            lw=1.5, mutation_scale=12))
    s = 40 + (tasks - tasks.min()) / max(1e-9, tasks.max() - tasks.min()) * 200
    ax.scatter(up[:, 0], up[:, 1], s=s, marker="*", color="goldenrod",
               edgecolor="k", linewidths=0.5, zorder=4, label="Users ($\\propto$ demand)")
    for idx, ((ux, uy), demand) in enumerate(zip(up, tasks), start=1):
        ax.annotate(f"$U_{idx}$: {demand:.2f}", (ux, uy),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.scatter([0], [0], marker="o", s=70, color="k", zorder=6, label="Start")
    half = 10 / 2
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m)", fontsize=14); ax.set_ylabel("y (m)", fontsize=14)
    ax.tick_params(labelsize=12); ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="lower left", ncol=2)
    plt.tight_layout()
    _save(fig, out_dir, "trajectory_case")


def table_compute_cost():
    """Consolidated compute-cost table replacing the sigma/user-density compute panels.
    Columns are the extremes of the noise and user-density sweeps; rows are methods.
    Also emits the LaTeX table body directly for pasting into the paper."""
    cols = [("sigma", 0.05, "$\\sigma^2{=}0.05$"), ("sigma", 0.25, "$\\sigma^2{=}0.25$"),
            ("num_users", 2, "$K{=}2$"), ("num_users", 10, "$K{=}10$")]
    os.makedirs(TABLE_ROOT, exist_ok=True)
    md = ["| Method | " + " | ".join(c[2].replace("$", "").replace("\\", "") for c in cols) + " |",
          "|---|---|---|---|---|"]
    tex = []
    for tag, lab in TAG2LABEL.items():
        vals = []
        for param, val, _ in cols:
            df = load(tag, param, val)
            vals.append(df["total_training_time"].median() if df is not None else None)
        md.append(f"| {lab} | " + " | ".join(f"{v:.0f}" if v is not None else "--" for v in vals) + " |")
        tex.append(f"{lab} & " + " & ".join(f"{v:.0f}" if v is not None else "--" for v in vals) + r" \\")
    with open(os.path.join(TABLE_ROOT, "table_compute.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(TABLE_ROOT, "table_compute_body.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("saved table_compute.md + table_compute_body.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Regenerate one sweep panel from the existing measured CSVs.")
    args = parser.parse_args()
    if args.smoke:
        plot_sweep("area_size", stem_suffix="_smoke")
        smoke_pdf = os.path.join(FIG_ROOT, "area", "area_size_time_step_smoke.pdf")
        if not os.path.exists(smoke_pdf) or os.path.getsize(smoke_pdf) == 0:
            raise RuntimeError("Smoke-test figure was not created.")
        print("Smoke test PASSED")
    else:
        for param in SWEEPS:
            plot_sweep(param)
            table_sweep(param)
        table_compute_cost()
        plot_alloc()
        plot_energy()
        plot_trajectory()

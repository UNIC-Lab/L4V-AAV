"""
plot_extended_results.py

Builds the paper-ready figures and Markdown result tables for the extended-
model experiments (allocation ablation, MPC-vs-L4V, energy Pareto, and the
enriched multi-baseline noise-power sweep) from the real CSVs produced by
run_extended_experiments.py / run_rich_sigma_sweep.py.

Figures are saved as both PDF (for \\includegraphics in ref.tex) and PNG (for
quick review) under paper/figs/{alloc,mpc,energy,rich_sigma}/, matching the
three-panel convention already used for Figs. 2-4 (figs/area, figs/sigma,
figs/num_user). Tables are saved as Markdown under paper_assets/tables/, the
source of truth that the LaTeX tables in ref.tex are transcribed from.

With few trials per condition (n=3-5), box+strip plots are used for grouped
comparisons instead of violins: a KDE-smoothed violin over 3-5 points
misrepresents the sample as a continuous density it is not. The noise-power
sweep itself is a trend plot (mean +/- std per sigma, one line per method),
which is the right chart for showing whether performance is monotonic in
noise power, the reviewer-facing question this round is about.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import METHOD_COLORS, METHOD_ORDER

DATA_DIR = "results_extended/data"
FIG_ROOT = os.path.join("..", "..", "..", "manuscript-draft", "figs")
TABLE_ROOT = "paper_assets/tables"

COLORS = dict(METHOD_COLORS)
COLORS["L4V (learned allocation)"] = COLORS["L4V"]

METHOD_LABELS = {"l4v": "L4V", "ga": "GA", "dqn": "DQN", "a2c": "A2C", "ddpg": "DDPG", "mpc": "MPC"}
MAX_TRAJ = 500  # a run hitting this cap did not complete the mission


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, out_dir, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {os.path.join(out_dir, stem)}.{{pdf,png}}")


def _box_strip(ax, data_list, labels, colors, ylabel):
    bp = ax.boxplot(data_list, patch_artist=True, widths=0.5)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(colors.get(lab, "#888888"))
        patch.set_alpha(0.35)
    for i, d in enumerate(data_list):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.85, s=32, zorder=3, color=colors.get(labels[i], "#333333"))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11, rotation=12, ha="right")
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, axis="y", alpha=0.25)


def _fmt(mean, std):
    if pd.isna(mean):
        return "--"
    return f"{mean:.3g} $\\pm$ {std:.2g}" if not pd.isna(std) else f"{mean:.3g}"


def _write_md_table(path, headers, rows, best_row_label=None, direction=None):
    """direction: dict metric_index(1-based col) -> 'min' or 'max' for bolding the best value."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    numeric_cols = {}
    if direction:
        for col, d in direction.items():
            vals = []
            for r in rows:
                try:
                    vals.append(float(str(r[col]).split()[0]))
                except (ValueError, IndexError):
                    vals.append(np.nan)
            best_idx = int(np.nanargmin(vals)) if d == "min" else int(np.nanargmax(vals))
            numeric_cols[col] = best_idx
    for i, r in enumerate(rows):
        cells = list(r)
        for col, best_idx in numeric_cols.items():
            if i == best_idx:
                cells[col] = f'**<span style="color:#d62728">{cells[col]}</span>**'
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Part A: allocation ablation
# ---------------------------------------------------------------------------
def plot_alloc_ablation():
    out_dir = _ensure_dir(os.path.join(FIG_ROOT, "alloc"))
    files = glob.glob(os.path.join(DATA_DIR, "alloc_ablation_users*_alloc*.csv"))
    if not files:
        print("No allocation-ablation CSVs found yet.")
        return

    users_set = sorted({int(os.path.basename(f).split("users")[1].split("_")[0]) for f in files})
    table_rows = []

    for metric, ylabel, stem in [
        ("avg_transmission_rate", "Average Transmission Rate", "alloc_avg_rate"),
        ("time_steps", "Mission Completion Time Steps", "alloc_time_step"),
        ("total_energy", "Total Flight Energy", "alloc_energy"),
    ]:
        fig, axes = plt.subplots(1, len(users_set), figsize=(6 * len(users_set), 6), squeeze=False)
        for j, num_users in enumerate(users_set):
            data_list, labels = [], []
            for alloc_flag, label in [(True, "L4V (learned allocation)"), (False, "L4V (equal split)")]:
                path = os.path.join(DATA_DIR, f"alloc_ablation_users{num_users}_alloc{alloc_flag}.csv")
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                data_list.append(df[metric].dropna().values)
                labels.append(label)
                if metric == "time_steps":
                    table_rows.append([num_users, label, f"{df[metric].mean():.1f}",
                                        f"{df[metric].std():.1f}", f"{df['total_energy'].mean():.1f}",
                                        f"{df['avg_transmission_rate'].mean():.3f}"])
            _box_strip(axes[0][j], data_list, labels, COLORS, ylabel if j == 0 else "")
            axes[0][j].set_xlabel(f"K = {num_users} users", fontsize=13)
        plt.tight_layout()
        _save(fig, out_dir, stem)

    _write_md_table(
        os.path.join(TABLE_ROOT, "table_alloc_ablation.md"),
        ["K", "Policy", "Time steps (mean)", "Time steps (std)", "Total energy (mean)", "Avg rate (mean)"],
        table_rows,
    )


# ---------------------------------------------------------------------------
# Part B: MPC vs L4V over sigma (kept for backward compatibility / quick check)
# ---------------------------------------------------------------------------
def plot_mpc_vs_l4v():
    out_dir = _ensure_dir(os.path.join(FIG_ROOT, "mpc"))
    files = glob.glob(os.path.join(DATA_DIR, "mpc_vs_l4v_*_sigma*.csv"))
    if not files:
        print("No MPC-vs-L4V CSVs found yet.")
        return

    sigmas = sorted({float(os.path.basename(f).split("sigma")[1].replace(".csv", "")) for f in files})

    for metric, ylabel, stem in [
        ("avg_transmission_rate", "Average Transmission Rate", "mpc_avg_rate"),
        ("time_steps", "Mission Completion Time Steps", "mpc_time_step"),
        ("total_training_time", "Compute Time to Deploy (s)", "mpc_compute_time"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 6))
        for method, label in [("l4v", "L4V"), ("mpc", "MPC")]:
            means, stds, xs = [], [], []
            for sigma in sigmas:
                path = os.path.join(DATA_DIR, f"mpc_vs_l4v_{method}_sigma{sigma}.csv")
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                vals = df[metric].dropna().values
                if len(vals) == 0:
                    continue
                xs.append(sigma)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            if xs:
                ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4, linewidth=2,
                             label=label, color=COLORS.get(label, None))
        ax.set_xlabel(r"Noise power $\sigma^2$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        _save(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# Part D: enriched multi-baseline sigma sweep (L4V, GA, DQN, A2C, DDPG, MPC)
# ---------------------------------------------------------------------------
def plot_rich_sigma_sweep():
    out_dir = _ensure_dir(os.path.join(FIG_ROOT, "rich_sigma"))
    files = glob.glob(os.path.join(DATA_DIR, "rich_sigma_*_sigma*.csv"))
    files = [f for f in files if "_smoke" not in f]
    if not files:
        print("No rich-sigma-sweep CSVs found yet.")
        return

    methods_present = sorted({os.path.basename(f).split("rich_sigma_")[1].split("_sigma")[0] for f in files},
                              key=lambda m: METHOD_ORDER.index(METHOD_LABELS.get(m, m))
                              if METHOD_LABELS.get(m, m) in METHOD_ORDER else 99)
    sigmas = sorted({float(os.path.basename(f).split("_sigma")[-1].replace(".csv", "")) for f in files})

    stats = {}  # method -> metric -> (means, stds, convergence_rate) aligned to sigmas
    for method in methods_present:
        stats[method] = {}
        for metric in ["avg_transmission_rate", "time_steps", "total_training_time"]:
            means, stds, conv_rates = [], [], []
            for sigma in sigmas:
                path = os.path.join(DATA_DIR, f"rich_sigma_{method}_sigma{sigma}.csv")
                if not os.path.exists(path):
                    means.append(np.nan); stds.append(np.nan); conv_rates.append(np.nan)
                    continue
                df = pd.read_csv(path)
                total = len(df)
                # Applied uniformly to every method, not just the offline-trained ones: a run
                # that hits the 500-step cap did not complete the mission regardless of which
                # algorithm produced it, so it is excluded from the trend statistics the same way.
                conv = df[df["time_steps"] < MAX_TRAJ]
                conv_rates.append(len(conv) / total if total else np.nan)
                use = conv if len(conv) > 0 else df
                means.append(use[metric].mean())
                stds.append(use[metric].std())
            stats[method][metric] = (np.array(means), np.array(stds), np.array(conv_rates))

    metric_specs = [
        ("avg_transmission_rate", "Average Transmission Rate", "rich_sigma_avg_rate"),
        ("time_steps", "Mission Completion Time Steps", "rich_sigma_time_step"),
        ("total_training_time", "Compute Time to Deploy (s)", "rich_sigma_compute_time"),
    ]
    for metric, ylabel, stem in metric_specs:
        fig, ax = plt.subplots(figsize=(6, 6))
        for method in methods_present:
            label = METHOD_LABELS.get(method, method)
            means, stds, conv = stats[method][metric]
            valid = ~np.isnan(means)
            ax.errorbar(np.array(sigmas)[valid], means[valid], yerr=stds[valid], marker="o",
                        capsize=4, linewidth=2, label=label, color=COLORS.get(label, None))
        ax.set_xlabel(r"Noise power $\sigma^2$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=11, ncol=2)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        _save(fig, out_dir, stem)

    # Mission-completion-rate figure, applied uniformly to all six methods: makes any
    # budget- or noise-driven non-monotonicity legible as its own explicit quantity
    # rather than left implicit in wide error bars on the trend plots above.
    # With n=3 trials, this rate is quantized to {0, 33, 67, 100}%, so different
    # methods frequently land on the exact same value and their lines would otherwise
    # sit exactly on top of each other; a small per-method vertical offset and distinct
    # marker shapes/line styles keep every series visible without altering the values
    # (offsets are disclosed here and in the caption).
    markers = ["o", "s", "^", "D", "v", "P"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, method in enumerate(methods_present):
        label = METHOD_LABELS.get(method, method)
        _, _, conv = stats[method]["time_steps"]
        jitter = (k - (len(methods_present) - 1) / 2) * 1.2
        ax.plot(sigmas, conv * 100 + jitter, marker=markers[k % len(markers)],
                linestyle=linestyles[k % len(linestyles)], linewidth=2, markersize=7,
                label=label, color=COLORS.get(label, None))
    ax.set_xlabel(r"Noise power $\sigma^2$", fontsize=14)
    ax.set_ylabel("Mission Completion Rate (%, jittered $\\pm$3pp to separate ties)", fontsize=12)
    ax.set_ylim(-8, 108)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, out_dir, "rich_sigma_convergence_rate")

    # Markdown table: one row per (method, sigma), plus a per-sigma gain row for L4V
    # over the strongest baseline (excluding MPC, which is not a learned baseline).
    rows = []
    for si, sigma in enumerate(sigmas):
        block = []
        baseline_times = {}
        for method in methods_present:
            label = METHOD_LABELS.get(method, method)
            r_mean, r_std, _ = stats[method]["avg_transmission_rate"]
            t_mean, t_std, conv = stats[method]["time_steps"]
            block.append([sigma, label, _fmt(r_mean[si], r_std[si]), _fmt(t_mean[si], t_std[si]),
                          f"{conv[si]*100:.0f}%" if not np.isnan(conv[si]) else "--"])
            if label not in ("L4V", "MPC") and not np.isnan(t_mean[si]):
                baseline_times[label] = t_mean[si]
        rows.extend(block)
        if baseline_times and "l4v" in stats:
            l4v_time = stats["l4v"]["time_steps"][0][si]
            best_baseline_label = min(baseline_times, key=baseline_times.get)
            best_baseline_time = baseline_times[best_baseline_label]
            if not np.isnan(l4v_time) and best_baseline_time > 0:
                gain_pct = (best_baseline_time - l4v_time) / best_baseline_time * 100
                rows.append([sigma, f"Gain over best baseline ({best_baseline_label})",
                            "--", f"{gain_pct:+.0f}% time", "--"])

    _write_md_table(
        os.path.join(TABLE_ROOT, "table_rich_sigma_sweep.md"),
        ["sigma^2", "Method", "Avg rate", "Time steps", "Converged"],
        rows,
    )
    print("[rich sigma] convergence rates by method (rows=sigma, cols=method):")
    conv_table = pd.DataFrame({METHOD_LABELS.get(m, m): stats[m]["time_steps"][2] for m in methods_present},
                               index=sigmas)
    print(conv_table)


# ---------------------------------------------------------------------------
# Part C: energy-time Pareto
# ---------------------------------------------------------------------------
def plot_energy_pareto():
    out_dir = _ensure_dir(os.path.join(FIG_ROOT, "energy"))
    files = glob.glob(os.path.join(DATA_DIR, "energy_pareto_gamma*.csv"))
    if not files:
        print("No energy-Pareto CSVs found yet.")
        return

    gammas = sorted({float(os.path.basename(f).split("gamma")[1].replace(".csv", "")) for f in files})

    energy_means, energy_stds, time_means, time_stds, rate_means, rate_stds, n_converged, n_total = \
        [], [], [], [], [], [], [], []
    for gamma in gammas:
        df = pd.read_csv(os.path.join(DATA_DIR, f"energy_pareto_gamma{gamma}.csv"))
        n_total.append(len(df))
        conv = df[df["time_steps"] < MAX_TRAJ]
        n_converged.append(len(conv))
        energy_means.append(conv["total_energy"].mean())
        energy_stds.append(conv["total_energy"].std())
        time_means.append(conv["time_steps"].mean())
        time_stds.append(conv["time_steps"].std())
        rate_means.append(conv["avg_transmission_rate"].mean())
        rate_stds.append(conv["avg_transmission_rate"].std())
        if len(conv) < len(df):
            print(f"  [energy pareto] gamma={gamma}: excluded {len(df) - len(conv)}/{len(df)} "
                  f"non-converged (500-step-capped) run(s) from statistics")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(time_means, energy_means, xerr=time_stds, yerr=energy_stds,
                fmt="o-", color=METHOD_COLORS["L4V"], capsize=4, linewidth=2, markersize=8)
    for g, x, y, nc, nt in zip(gammas, time_means, energy_means, n_converged, n_total):
        label = f"$\\gamma$={g:g}" + (f" ({nc}/{nt})" if nc < nt else "")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.set_xlabel("Mission Completion Time Steps", fontsize=14)
    ax.set_ylabel("Total Flight Energy", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, out_dir, "energy_time_pareto")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(gammas, rate_means, yerr=rate_stds, marker="o", color=METHOD_COLORS["DDPG"],
                capsize=4, linewidth=2)
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlabel(r"Energy weight $\gamma$", fontsize=14)
    ax.set_ylabel("Average Transmission Rate", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, out_dir, "energy_rate_tradeoff")

    table_rows = [[g, f"{tm:.1f}" if not pd.isna(tm) else "--", f"{em:.1f}" if not pd.isna(em) else "--",
                    f"{rm:.3f}" if not pd.isna(rm) else "--", f"{nc}/{nt}"]
                   for g, tm, em, rm, nc, nt in zip(gammas, time_means, energy_means, rate_means,
                                                      n_converged, n_total)]
    _write_md_table(
        os.path.join(TABLE_ROOT, "table_energy_pareto.md"),
        ["gamma", "Time steps (mean)", "Total energy (mean)", "Avg rate (mean)", "Converged/Total"],
        table_rows,
    )


if __name__ == "__main__":
    plot_alloc_ablation()
    plot_mpc_vs_l4v()
    plot_energy_pareto()
    plot_rich_sigma_sweep()

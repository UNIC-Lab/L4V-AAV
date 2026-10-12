"""Render the second-round allocation figure with a positive median focus."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results_extended" / "data"
OUTPUT = ROOT.parents[2] / "manuscript-round2" / "figs" / "alloc"
OUTPUT.mkdir(parents=True, exist_ok=True)


def plot(metric, ylabel, stem, ticks):
    fig, ax = plt.subplots(figsize=(6.4, 4.15))
    positions = []
    labels = []
    for group, users in enumerate((4, 8)):
        equal = pd.read_csv(DATA / f"allocfull_users{users}_allocFalse.csv")[metric].to_numpy(float)
        learned = pd.read_csv(DATA / f"allocfull_users{users}_allocTrue.csv")[metric].to_numpy(float)
        x0, x1 = 3 * group, 3 * group + 1
        for left, right in zip(equal, learned):
            ax.plot([x0, x1], [left, right], color="#9ca3af", linewidth=1.0, alpha=0.65, zorder=1)
        ax.scatter(np.full_like(equal, x0), equal, s=34, facecolor="white", edgecolor="#4c78a8", zorder=2)
        ax.scatter(np.full_like(learned, x1), learned, s=38, color="#d62728", edgecolor="white", linewidth=0.5, zorder=2)
        med_equal = float(np.median(equal))
        med_learned = float(np.median(learned))
        ax.plot([x0 - .24, x0 + .24], [med_equal] * 2, color="#4c78a8", linewidth=4, solid_capstyle="round")
        ax.plot([x1 - .24, x1 + .24], [med_learned] * 2, color="#d62728", linewidth=4, solid_capstyle="round")
        reduction = 100 * (med_equal - med_learned) / med_equal
        ax.text((x0 + x1) / 2, ticks[-1] / 1.07, f"median reduction {reduction:.0f}%",
                ha="center", va="top", fontsize=9, color="#b91c1c")
        positions.extend([x0, x1])
        labels.extend([f"Equal\n$K={users}$", f"Learned\n$K={users}$"])
    ax.set_yscale("log")
    ax.set_ylim(ticks[0] / 1.08, ticks[-1] * 1.05)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{value:g}" for value in ticks])
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb", linewidth=.7, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], s=34, facecolor="white", edgecolor="#4c78a8", label="Equal split")
    ax.scatter([], [], s=38, color="#d62728", label="Learned allocation")
    ax.plot([], [], color="#9ca3af", label="Paired layout")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=8)
    fig.tight_layout(pad=.5)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot("time_steps", "Completion time (slots)", "alloc_time_step", [20, 40, 80, 160, 320, 500])
    plot("total_energy", "Total flight energy", "alloc_energy", [400, 800, 1600, 3200, 6400, 9000])

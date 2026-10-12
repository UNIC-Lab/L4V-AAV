"""Build second-round manuscript figures from persisted experiment arrays."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_round2"
FIGURES = ROOT.parents[2] / "manuscript-round2" / "figs" / "round2"


def paired_plot(values, gammas, ylabel, output, color="#d62728"):
    unique = np.unique(gammas)
    if len(unique) != 2:
        raise ValueError("The paired high-speed figure expects two registered energy weights")
    groups = [values[gammas == gamma] for gamma in unique]
    if len(groups[0]) != len(groups[1]):
        raise ValueError("High-speed arms are not paired")
    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    for left, right in zip(groups[0], groups[1]):
        ax.plot([0, 1], [left, right], color="#9ca3af", linewidth=0.8, alpha=0.8, zorder=1)
        ax.scatter([0, 1], [left, right], color=["#4c78a8", color], s=18, zorder=2)
    medians = [np.median(group) for group in groups]
    ax.plot([0, 1], medians, color="#111827", linewidth=2.2, marker="o", markersize=5, zorder=3)
    ax.set_xticks([0, 1], [r"$\gamma=0$", r"$\gamma=10^{-3}$"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURES / f"{output}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{output}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    with np.load(RESULTS / "high_speed" / "high_speed_formal.npz", allow_pickle=True) as raw:
        gamma = raw["gamma"].astype(float)
        completion = raw["time_steps"].astype(float)
        energy = raw["total_energy_j"].astype(float)
        mean_speed = raw["mean_speed_mps"].astype(float)
        p95_speed = raw["p95_speed_mps"].astype(float)
    paired_plot(completion, gamma, "Completion time (slots)", "high_speed_completion")
    paired_plot(energy, gamma, "Flight energy (J)", "high_speed_energy")

    summary = {
        "high_speed": [],
        "surrogate_gap": json.loads((RESULTS / "surrogate_gap" / "summary_formal.json").read_text(encoding="utf-8")),
        "qcqp": json.loads((RESULTS / "qcqp" / "summary_formal.json").read_text(encoding="utf-8")),
    }
    for value in np.unique(gamma):
        mask = gamma == value
        summary["high_speed"].append({
            "gamma": float(value),
            "median_completion_slots": float(np.median(completion[mask])),
            "median_energy_j": float(np.median(energy[mask])),
            "median_mean_speed_mps": float(np.median(mean_speed[mask])),
            "median_p95_speed_mps": float(np.median(p95_speed[mask])),
            "completion_fraction": float(np.mean(completion[mask] < 500)),
        })
    (RESULTS / "round2_verified_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

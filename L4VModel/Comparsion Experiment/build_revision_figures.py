"""Build high-information-density revision figures from saved experiment data.

The script only reads completed per-trial CSV records. It does not rerun or
modify any experiment. Each panel is saved independently so that LaTeX can
compose the final multi-panel narrative figure without rasterizing labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


METHOD_COLOR = "#d62728"
PAIRED_COLOR = "#4C78A8"
ENERGY_MARKER_COLOR = "#355C7D"
GAMMAS = [0.0, 1e-5, 1e-4, 3e-4, 1e-3]
GAMMA_LABELS = ["0", r"10^{-5}", r"10^{-4}", r"3{\times}10^{-4}", r"10^{-3}"]
REQUIRED_COLUMNS = {"run_index", "time_steps", "total_energy"}


def load_records(data_dir: Path, smoke: bool) -> pd.DataFrame:
    gammas = GAMMAS[:2] if smoke else GAMMAS
    frames: list[pd.DataFrame] = []
    for gamma in gammas:
        path = data_dir / f"energyfull_gamma{gamma}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing source data: {path}")
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame = frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()
        frame["gamma"] = gamma
        frames.append(frame)

    records = pd.concat(frames, ignore_index=True)
    if smoke:
        records = records[records["run_index"] <= 2].copy()
    if records.empty or not np.isfinite(records[["time_steps", "total_energy"]].to_numpy()).all():
        raise ValueError("The saved experiment records are empty or contain non-finite values.")
    return records


def percentile_summary(values: np.ndarray) -> tuple[float, float, float]:
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return float(q25), float(median), float(q75)


def save_figure(fig: plt.Figure, stem: str, output_dirs: list[Path]) -> None:
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)


def plot_pareto(records: pd.DataFrame, output_dirs: list[Path], suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    gammas = sorted(records["gamma"].unique())

    ax.scatter(
        records["time_steps"],
        records["total_energy"],
        facecolor="white",
        edgecolor=PAIRED_COLOR,
        linewidth=1.0,
        s=30,
        alpha=0.72,
        label="Paired trials",
        zorder=2,
    )

    median_time: list[float] = []
    median_energy: list[float] = []
    for gamma in gammas:
        group = records[records["gamma"] == gamma]
        tq25, tmed, tq75 = percentile_summary(group["time_steps"].to_numpy())
        eq25, emed, eq75 = percentile_summary(group["total_energy"].to_numpy())
        median_time.append(tmed)
        median_energy.append(emed)
        ax.errorbar(
            tmed,
            emed,
            xerr=[[tmed - tq25], [tq75 - tmed]],
            yerr=[[emed - eq25], [eq75 - emed]],
            fmt="D",
            markersize=7,
            color=METHOD_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.6,
            capsize=4,
            linewidth=1.8,
            zorder=4,
        )

    ax.plot(
        median_time,
        median_energy,
        color=METHOD_COLOR,
        linewidth=2.0,
        label="Median across trials",
        zorder=3,
    )
    for gamma_to_label, offset in [(0.0, (8, 8)), (1e-4, (-12, -24))]:
        if gamma_to_label not in gammas:
            continue
        index = gammas.index(gamma_to_label)
        ax.annotate(
            rf"$\gamma={GAMMA_LABELS[GAMMAS.index(gamma_to_label)]}$",
            (median_time[index], median_energy[index]),
            xytext=offset,
            textcoords="offset points",
            fontsize=11,
        )

    slope, intercept = np.polyfit(records["time_steps"], records["total_energy"], deg=1)
    x_fit = np.linspace(records["time_steps"].min(), records["time_steps"].max(), 200)
    y_fit = slope * x_fit + intercept
    residual = records["total_energy"] - (slope * records["time_steps"] + intercept)
    total = records["total_energy"] - records["total_energy"].mean()
    r_squared = 1 - float(np.sum(residual**2) / np.sum(total**2))
    ax.plot(
        x_fit,
        y_fit,
        color="#444444",
        linestyle="--",
        linewidth=1.3,
        label=rf"Linear fit, $R^2={r_squared:.3f}$",
        zorder=1,
    )
    ax.annotate(
        "Lower is better",
        xy=(records["time_steps"].min(), records["total_energy"].min()),
        xytext=(28, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 1.0},
        fontsize=11,
        color="#444444",
    )
    ax.set_xlabel("Mission completion time (steps)", fontsize=14)
    ax.set_ylabel("Total flight energy (normalized units)", fontsize=14)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(labelsize=12, which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=3)
    ax.grid(True, alpha=0.22, which="major", linewidth=0.8)
    ax.grid(True, alpha=0.08, which="minor", linewidth=0.5)
    ax.legend(
        fontsize=10.5,
        frameon=False,
        loc="upper left",
        borderaxespad=0.35,
        handlelength=2.1,
        labelspacing=0.35,
    )
    ax.text(0.985, 0.86, "(c)", transform=ax.transAxes, ha="right",
            va="top", fontsize=11, fontweight="bold")
    x_min, x_max = records["time_steps"].min(), records["time_steps"].max()
    y_min, y_max = records["total_energy"].min(), records["total_energy"].max()
    ax.set_xlim(x_min - 0.035 * (x_max - x_min), x_max + 0.035 * (x_max - x_min))
    ax.set_ylim(y_min - 0.045 * (y_max - y_min), y_max + 0.045 * (y_max - y_min))
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.175, top=0.975)
    save_figure(fig, f"energy_tradeoff_pareto{suffix}", output_dirs)
    plt.close(fig)


def plot_normalized(records: pd.DataFrame, output_dirs: list[Path], suffix: str) -> None:
    baseline = records[records["gamma"] == 0.0].set_index("run_index")
    paired = records.join(
        baseline[["time_steps", "total_energy"]],
        on="run_index",
        rsuffix="_baseline",
    )
    paired["time_change"] = 100 * (
        paired["time_steps"] / paired["time_steps_baseline"] - 1
    )
    paired["energy_change"] = 100 * (
        paired["total_energy"] / paired["total_energy_baseline"] - 1
    )

    gammas = sorted(records["gamma"].unique())
    x_values = np.arange(len(gammas), dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    plot_specs = [
        ("time_change", "Completion time", "o", "-", METHOD_COLOR, -0.055),
        ("energy_change", "Flight energy", "s", "--", ENERGY_MARKER_COLOR, 0.055),
    ]
    time_medians: np.ndarray | None = None
    for column, label, marker, linestyle, color, offset in plot_specs:
        medians: list[float] = []
        q25_values: list[float] = []
        q75_values: list[float] = []
        for gamma in gammas:
            values = paired.loc[paired["gamma"] == gamma, column].to_numpy()
            q25, median, q75 = percentile_summary(values)
            q25_values.append(q25)
            medians.append(median)
            q75_values.append(q75)
        for run_index in sorted(paired["run_index"].unique()):
            run_values = (
                paired[paired["run_index"] == run_index]
                .set_index("gamma")
                .reindex(gammas)[column]
                .to_numpy()
            )
            ax.plot(
                x_values + offset,
                run_values,
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=0.18,
                zorder=1,
            )
        medians_array = np.asarray(medians)
        if column == "time_change":
            time_medians = medians_array.copy()
        ax.plot(
            x_values + offset,
            medians_array,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=2.2,
            markersize=7,
            label=label,
            zorder=3,
        )
        ax.fill_between(
            x_values + offset,
            np.asarray(q25_values),
            np.asarray(q75_values),
            color=color,
            alpha=0.16,
            zorder=2,
        )

    ax.axhline(0, color="#555555", linestyle=":", linewidth=1.2)
    ax.set_xticks(x_values)
    ax.set_xticklabels([rf"${GAMMA_LABELS[GAMMAS.index(g)]}$" for g in gammas])
    ax.set_xlabel(r"Energy weight $\gamma$", fontsize=14)
    ax.set_ylabel(r"Change from paired $\gamma=0$ (\%)", fontsize=14)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(labelsize=12, which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=3)
    ax.grid(True, axis="y", alpha=0.22, which="major", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.08, which="minor", linewidth=0.5)
    ax.legend(
        fontsize=10.5,
        frameon=False,
        loc="upper left",
        borderaxespad=0.35,
        handlelength=2.1,
        labelspacing=0.35,
    )
    ax.text(0.985, 0.985, "(d)", transform=ax.transAxes, ha="right",
            va="top", fontsize=11, fontweight="bold")
    if 1e-4 in gammas and time_medians is not None:
        gamma_index = gammas.index(1e-4)
        improved = int(
            np.sum(
                paired.loc[paired["gamma"] == 1e-4, "time_change"].to_numpy() < 0
            )
        )
        ax.annotate(
            f"{improved}/5 improve",
            (x_values[gamma_index] - 0.055, time_medians[gamma_index]),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=METHOD_COLOR,
        )
    all_changes = paired[["time_change", "energy_change"]].to_numpy()
    y_min, y_max = float(all_changes.min()), float(all_changes.max())
    y_pad = 0.04 * (y_max - y_min)
    ax.set_xlim(-0.28, len(gammas) - 1 + 0.28)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.175, top=0.975)
    save_figure(fig, f"energy_tradeoff_normalized{suffix}", output_dirs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two paired trials and the first two saved gamma settings.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "results_extended" / "data"
    paper_assets_dir = script_dir / "paper_assets" / "figures"
    manuscript_dir = script_dir.parents[3] / "paper" / "figs" / "energy"
    suffix = "_smoke" if args.smoke else ""
    output_dirs = [paper_assets_dir, manuscript_dir]

    records = load_records(data_dir, smoke=args.smoke)
    plot_pareto(records, output_dirs, suffix)
    plot_normalized(records, output_dirs, suffix)

    if args.smoke:
        print("Smoke test PASSED")
    else:
        print(f"Saved revision figures to {paper_assets_dir} and {manuscript_dir}")


if __name__ == "__main__":
    main()

"""Create publication figures from the executed L4V stress-test NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import AutoMinorLocator, NullFormatter

from style import METHOD_COLORS


L4V_RED = METHOD_COLORS["L4V"]
PAIR_GRAY = "#A7A7A7"
BLUE = "#355C7D"
ORANGE = "#E68613"
OBSTACLE_CLEARANCE_MARGIN = 0.20


def source_paths(data_dir: Path, suffix: str) -> dict[str, Path]:
    if suffix in ("_smoke", "_validate"):
        common = data_dir / f"stress_suite{suffix}.npz"
        return {
            part: common
            for part in ("robustness", "multi", "obstacle", "smoothness")
        }
    return {
        part: data_dir / f"stress_suite_{part}{suffix}.npz"
        for part in ("robustness", "multi", "obstacle", "smoothness")
    }


def load_sources(data_dir: Path, suffix: str) -> dict[str, np.lib.npyio.NpzFile]:
    paths = source_paths(data_dir, suffix)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing stress-test data: " + ", ".join(missing))
    return {
        name: np.load(path, allow_pickle=True) for name, path in paths.items()
    }


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=11, which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=3)
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.8)
    ax.grid(True, axis="y", which="minor", alpha=0.07, linewidth=0.5)


def paired_condition_plot(
    values: np.ndarray,
    success: np.ndarray,
    labels: list[str],
    panel: str,
    ylabel: str,
) -> plt.Figure:
    per_trial = np.median(values, axis=2)
    normalized = 100.0 * per_trial / per_trial[:, [0]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for trial in range(normalized.shape[0]):
        ax.plot(
            x,
            normalized[trial],
            color=PAIR_GRAY,
            marker="o",
            markersize=4,
            linewidth=1.0,
            alpha=0.65,
            zorder=1,
        )
    median = np.median(normalized, axis=0)
    q25, q75 = np.percentile(normalized, [25, 75], axis=0)
    ax.plot(
        x,
        median,
        color=L4V_RED,
        marker="D",
        markersize=7,
        linewidth=2.4,
        label="Median normalized completion",
        zorder=3,
    )
    ax.fill_between(x, q25, q75, color=L4V_RED, alpha=0.15, zorder=2)
    ax.axhline(100, color="#555555", linestyle=":", linewidth=1.1)
    success_rate = 100.0 * np.mean(success, axis=(0, 2))
    for index, rate in enumerate(success_rate):
        ax.annotate(
            f"{rate:.0f}% success",
            (x[index], median[index]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color=BLUE,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(-0.28, len(labels) - 1 + 0.28)
    span = max(10.0, float(np.max(normalized) - np.min(normalized)))
    observed_min = float(np.min(normalized))
    if observed_min < 70.0:
        ax.set_yscale("log")
        ax.set_ylim(max(5.0, observed_min * 0.75), float(np.max(normalized) * 1.28))
        ax.set_yticks([10, 20, 50, 100])
        ax.set_yticklabels(["10", "20", "50", "100"])
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(
            max(70.0, observed_min - 0.10 * span),
            float(np.max(normalized) + 0.16 * span),
        )
    style_axis(ax)
    ax.legend(
        loc="lower left" if observed_min < 70.0 else "upper left",
        fontsize=10,
        frameon=False,
        borderaxespad=0.35,
        handlelength=2.0,
    )
    ax.text(
        0.985,
        0.985,
        panel,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.17, top=0.975)
    return fig


def plot_multi(data: np.lib.npyio.NpzFile) -> plt.Figure:
    steps = np.median(data["multi_steps"], axis=2)
    distance = np.median(data["multi_fleet_distance"], axis=2)
    x = np.array([0.0, 1.0])
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for trial in range(steps.shape[0]):
        ax.plot(
            x - 0.035,
            steps[trial],
            color="#E59A98",
            marker="o",
            markerfacecolor="white",
            markersize=5,
            linewidth=1.0,
            alpha=0.75,
        )
    step_median = np.median(steps, axis=0)
    ax.plot(
        x - 0.035,
        step_median,
        color=L4V_RED,
        marker="o",
        markersize=8,
        linewidth=2.6,
        label="Completion time",
        zorder=3,
    )
    ax2 = ax.twinx()
    for trial in range(distance.shape[0]):
        ax2.plot(
            x + 0.035,
            distance[trial],
            color="#9FB2C4",
            marker="s",
            markerfacecolor="white",
            markersize=5,
            linewidth=1.0,
            linestyle="--",
            alpha=0.70,
        )
    distance_median = np.median(distance, axis=0)
    ax2.plot(
        x + 0.035,
        distance_median,
        color=BLUE,
        marker="s",
        markersize=7,
        linewidth=2.2,
        linestyle="--",
        label="Fleet path length",
        zorder=3,
    )
    reduction = 100.0 * (1.0 - step_median[1] / step_median[0])
    success = 100.0 * np.mean(data["multi_success"], axis=(0, 2))
    ax.annotate(
        f"median {reduction:.0f}% faster\n{success[1]:.0f}% success",
        (1.0, step_median[1]),
        xytext=(-22, 18),
        textcoords="offset points",
        ha="center",
        color=L4V_RED,
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["One AAV", "Two AAVs"])
    ax.set_ylabel("Mission completion time (steps)", fontsize=13, color=L4V_RED)
    ax2.set_ylabel("Fleet path length (normalized)", fontsize=13, color=BLUE)
    ax.tick_params(axis="y", labelcolor=L4V_RED)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax.set_xlim(-0.28, 1.28)
    style_axis(ax)
    ax2.tick_params(labelsize=11, direction="in", right=True)
    lines = ax.get_lines()[-1:] + ax2.get_lines()[-1:]
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        fontsize=10,
        frameon=False,
        borderaxespad=0.35,
    )
    ax.text(
        0.985,
        0.985,
        "(c)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.17, top=0.975)
    return fig


def plot_obstacle(data: np.lib.npyio.NpzFile) -> plt.Figure:
    trajectories = data["obstacle_trajectories"]
    scenarios = data["obstacle_scenarios"]
    clearance_gap = (
        np.median(data["obstacle_clearances"][:, 1], axis=1)
        - np.median(data["obstacle_clearances"][:, 0], axis=1)
    )
    trial = int(np.argmax(clearance_gap))
    scenario = scenarios[trial].item() if hasattr(scenarios[trial], "item") else scenarios[trial]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    colours = [ORANGE, L4V_RED]
    labels = ["Obstacle-unaware", "Obstacle-aware"]
    linestyles = ["--", "-"]
    for method in range(2):
        path = np.asarray(trajectories[trial, method])
        if path.ndim == 3:
            path = path[:, 0, :]
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=colours[method],
            linestyle=linestyles[method],
            linewidth=2.1,
            label=labels[method],
            zorder=3,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            color=colours[method],
            s=34,
            zorder=4,
        )
    for obstacle in scenario["obstacles"]:
        ax.add_patch(
            Circle(
                obstacle[:2],
                obstacle[2] + OBSTACLE_CLEARANCE_MARGIN,
                facecolor="none",
                edgecolor=L4V_RED,
                linestyle=":",
                alpha=0.75,
                linewidth=1.2,
                label="Training buffer",
                zorder=2,
            )
        )
        ax.add_patch(
            Circle(
                obstacle[:2],
                obstacle[2],
                facecolor="#555555",
                edgecolor="black",
                alpha=0.30,
                linewidth=1.1,
                label="No-fly region",
                zorder=2,
            )
        )
    users = scenario["users"]
    tasks = scenario["tasks"]
    ax.scatter(
        users[:, 0],
        users[:, 1],
        marker="*",
        s=50 + 25 * tasks,
        color=BLUE,
        edgecolor="black",
        linewidth=0.5,
        label="Users",
        zorder=5,
    )
    start = scenario["start"][0]
    ax.scatter(
        start[0],
        start[1],
        marker="o",
        s=55,
        color="black",
        label="Start",
        zorder=5,
    )
    safe_rate = 100.0 * np.mean(
        data["obstacle_safe_success"], axis=(0, 2)
    )
    median_clearance = np.median(
        data["obstacle_clearances"], axis=(0, 2)
    )
    summary = (
        f"Safe completion (U/A): {safe_rate[0]:.0f}% / {safe_rate[1]:.0f}%\n"
        f"Median clearance (U/A): {median_clearance[0]:.2f} / "
        f"{median_clearance[1]:.2f}"
    )
    ax.text(
        0.985,
        0.03,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
    )
    ax.set_xlabel("Horizontal position $x$ (normalized)", fontsize=13)
    ax.set_ylabel("Horizontal position $y$ (normalized)", fontsize=13)
    ax.set_xlim(-4.7, 5.0)
    ax.set_ylim(-4.1, 4.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=11, direction="in", top=True, right=True)
    handles, legend_labels = ax.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        fontsize=9.5,
        frameon=False,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.8,
    )
    ax.text(
        0.985,
        0.985,
        "(d)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.17, top=0.975)
    return fig


def plot_zero_shot(data: np.lib.npyio.NpzFile) -> plt.Figure:
    steps = data["zero_shot_steps"]
    per_layout = np.median(steps, axis=1)
    order = np.argsort(per_layout)
    sorted_steps = steps[order]
    ranks = np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for evaluation in range(sorted_steps.shape[1]):
        ax.scatter(
            ranks,
            sorted_steps[:, evaluation],
            s=21,
            facecolor="white",
            edgecolor=PAIR_GRAY,
            linewidth=0.8,
            alpha=0.8,
            label="Held-out rollout" if evaluation == 0 else None,
            zorder=1,
        )
    medians = np.median(sorted_steps, axis=1)
    ax.plot(
        ranks,
        medians,
        color=L4V_RED,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="Per-layout median",
        zorder=3,
    )
    overall = float(np.median(steps))
    q25, q75 = np.percentile(steps, [25, 75])
    ax.axhspan(q25, q75, color=L4V_RED, alpha=0.12, zorder=0)
    ax.axhline(overall, color=L4V_RED, linestyle="--", linewidth=1.4)
    success = 100.0 * np.mean(data["zero_shot_success"])
    layouts = int(data["pretrain_layout_count"])
    pretrain_seconds = float(data["pretrain_wall_time"])
    inference_ms = 1000.0 * float(
        np.median(data["zero_shot_inference_times"])
    )
    ax.text(
        0.03,
        0.96,
        f"Pretrained on {layouts:,} layouts\n"
        f"{success:.0f}% held-out success; median {overall:.0f} steps\n"
        f"One-off pretraining: {pretrain_seconds:.0f} s\n"
        f"Median frozen-policy rollout: {inference_ms:.1f} ms",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=BLUE,
    )
    ax.set_xlabel("Held-out layout rank", fontsize=13)
    ax.set_ylabel("Zero-shot completion time (steps)", fontsize=13)
    ax.set_xlim(0.4, len(order) + 0.6)
    if len(order) > 8:
        ax.set_xticks(np.linspace(1, len(order), 5, dtype=int))
    style_axis(ax)
    ax.legend(
        loc="lower right",
        fontsize=9.5,
        frameon=False,
        borderaxespad=0.35,
    )
    ax.text(
        0.985,
        0.985,
        "(e)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.17, top=0.975)
    return fig


def plot_smoothness(data: np.lib.npyio.NpzFile) -> plt.Figure:
    weights = data["smoothness_weights"]
    steps = np.median(data["smoothness_steps"], axis=2)
    turn = np.median(data["smoothness_turn_variation"], axis=2)
    x = np.arange(len(weights))
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for trial in range(steps.shape[0]):
        ax.plot(
            x - 0.025,
            steps[trial],
            color="#E59A98",
            marker="o",
            markerfacecolor="white",
            markersize=4.5,
            linewidth=0.9,
            alpha=0.68,
        )
    step_median = np.median(steps, axis=0)
    line_steps = ax.plot(
        x - 0.025,
        step_median,
        color=L4V_RED,
        marker="o",
        markersize=7,
        linewidth=2.4,
        label="Completion time",
        zorder=3,
    )[0]
    ax2 = ax.twinx()
    for trial in range(turn.shape[0]):
        ax2.plot(
            x + 0.025,
            turn[trial],
            color="#9FB2C4",
            marker="s",
            markerfacecolor="white",
            markersize=4.5,
            linewidth=0.9,
            linestyle="--",
            alpha=0.68,
        )
    turn_median = np.median(turn, axis=0)
    line_turn = ax2.plot(
        x + 0.025,
        turn_median,
        color=BLUE,
        marker="s",
        markersize=6.5,
        linewidth=2.1,
        linestyle="--",
        label="Turn variation",
        zorder=3,
    )[0]
    success = 100.0 * np.mean(data["smoothness_success"], axis=(0, 2))
    for index, rate in enumerate(success):
        ax.annotate(
            f"{rate:.0f}%",
            (x[index], step_median[index]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=L4V_RED,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{weight:g}" for weight in weights])
    ax.set_xlabel("Heading-smoothness weight $\\beta$", fontsize=13)
    ax.set_ylabel("Completion time (steps)", fontsize=13, color=L4V_RED)
    ax2.set_ylabel("Circular turn variation", fontsize=13, color=BLUE)
    # A logarithmic ordinate retains capped 500-step trials without allowing
    # them to compress the practically relevant 50--75-step median region.
    ax.set_yscale("log")
    ax.set_ylim(24, 540)
    ax.set_yticks([25, 50, 100, 250, 500])
    ax.set_yticklabels(["25", "50", "100", "250", "500"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", labelcolor=L4V_RED)
    ax2.tick_params(axis="y", labelcolor=BLUE, labelsize=11, direction="in")
    ax.set_xlim(-0.28, len(weights) - 0.72)
    style_axis(ax)
    ax.legend(
        [line_steps, line_turn],
        ["Completion time", "Turn variation"],
        loc="upper left",
        fontsize=9.5,
        frameon=False,
        borderaxespad=0.35,
    )
    ax.text(
        0.985,
        0.985,
        "(f)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.17, top=0.975)
    return fig


def write_smoothness_table(
    data: np.lib.npyio.NpzFile, output_dir: Path, suffix: str
) -> None:
    weights = data["smoothness_weights"]
    per_trial_steps = np.median(data["smoothness_steps"], axis=2)
    per_trial_turn = np.median(data["smoothness_turn_variation"], axis=2)
    success = 100.0 * np.mean(data["smoothness_success"], axis=(0, 2))
    step_median = np.median(per_trial_steps, axis=0)
    step_q25, step_q75 = np.percentile(per_trial_steps, [25, 75], axis=0)
    turn_median = np.median(per_trial_turn, axis=0)
    turn_change = 100.0 * (turn_median / turn_median[0] - 1.0)
    best_step = int(np.argmin(step_median))
    best_turn = int(np.argmin(turn_median))

    rows = [
        "| $\\beta$ | Completion steps, median [IQR] | Success | "
        "Turn variation | Change from $\\beta=0$ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, weight in enumerate(weights):
        step_text = (
            f"{step_median[index]:.0f} "
            f"[{step_q25[index]:.0f}, {step_q75[index]:.0f}]"
        )
        turn_text = f"{turn_median[index]:.4f}"
        if index == best_step:
            step_text = f"**{step_text}**"
        if index == best_turn:
            turn_text = f"**{turn_text}**"
        rows.append(
            f"| {weight:g} | {step_text} | {success[index]:.0f}% | "
            f"{turn_text} | {turn_change[index]:+.1f}% |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"smoothness_sensitivity{suffix}.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def save(fig: plt.Figure, stem: str, output_dirs: list[Path]) -> None:
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.validate:
        parser.error("--smoke and --validate are mutually exclusive")
    suffix = "_smoke" if args.smoke else "_validate" if args.validate else ""

    script_dir = Path(__file__).resolve().parent
    sources = load_sources(script_dir / "results_extended" / "stress", suffix)
    generalization_path = (
        script_dir
        / "results_extended"
        / "generalization"
        / f"zero_shot_generalization{suffix}.npz"
    )
    if not generalization_path.exists():
        raise FileNotFoundError(
            f"Missing zero-shot generalization data: {generalization_path}"
        )
    generalization = np.load(generalization_path, allow_pickle=True)
    paper_assets = script_dir / "paper_assets" / "figures"
    table_assets = script_dir / "paper_assets" / "tables"
    manuscript = script_dir.parents[3] / "paper" / "figs" / "stress"
    output_dirs = [paper_assets, manuscript]

    robustness = sources["robustness"]
    channel_labels = robustness["channel_condition_labels"].tolist()
    observation_labels = robustness["observation_condition_labels"].tolist()
    observation_display = [
        {
            "Full": "Full",
            "Location noise": "Location\nbias",
            "50% backlog mask": "50% backlog\nmask",
            "Both": "Bias +\nmask",
        }.get(label, label)
        for label in observation_labels
    ]
    save(
        paired_condition_plot(
            robustness["channel_steps"],
            robustness["channel_success"],
            channel_labels,
            "(a)",
            "Completion relative to matched model (%)",
        ),
        f"stress_channel_mismatch{suffix}",
        output_dirs,
    )
    save(
        paired_condition_plot(
            robustness["observation_steps"],
            robustness["observation_success"],
            observation_display,
            "(b)",
            "Completion relative to full observation (%)",
        ),
        f"stress_partial_observation{suffix}",
        output_dirs,
    )
    save(
        plot_multi(sources["multi"]),
        f"stress_multi_aav{suffix}",
        output_dirs,
    )
    save(
        plot_obstacle(sources["obstacle"]),
        f"stress_obstacle{suffix}",
        output_dirs,
    )
    save(
        plot_zero_shot(generalization),
        f"stress_zero_shot{suffix}",
        output_dirs,
    )
    save(
        plot_smoothness(sources["smoothness"]),
        f"stress_smoothness{suffix}",
        output_dirs,
    )
    write_smoothness_table(sources["smoothness"], table_assets, suffix)
    if args.smoke:
        print("Smoke test PASSED")
    else:
        print(f"Saved stress-test figures to {paper_assets} and {manuscript}")


if __name__ == "__main__":
    main()

"""Focused probes for CRN variance reduction and learned residual calibration."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import run_gradient_information_exploration as core


def crn_variance_probe(
    replicates: int,
    horizon: int,
    scenario_seed: int,
    policy_seed: int,
) -> dict[str, np.ndarray | float]:
    scenario = core.make_validation_scenario(scenario_seed, num_users=4)
    torch.manual_seed(policy_seed)
    policy = core.FleetPolicy(4, num_aavs=1, num_obstacles=0).to(core.DEVICE)
    models = core.build_ensemble(core.VARIANTS[4], policy_seed + 300_000)
    branches = core.make_action_branches(
        policy_seed * 100_000,
        3,
        horizon,
        scenario.tasks.numel(),
    )
    pair = [branches[0], branches[1]]
    independent_delta = np.zeros(replicates)
    crn_delta = np.zeros(replicates)

    with torch.no_grad():
        for replicate in range(replicates):
            seed = policy_seed * 10_000 + replicate
            shared = core.make_channel_noise(
                seed, horizon, 1, scenario.tasks.numel()
            )
            independent = [
                core.make_channel_noise(
                    seed + 100_000 + 101 * index,
                    horizon,
                    1,
                    scenario.tasks.numel(),
                )
                for index in range(2)
            ]
            crn_matrix, _ = core.branch_model_matrix(
                policy,
                scenario,
                pair,
                [shared, shared],
                horizon,
                models,
                keep_traces=False,
            )
            independent_matrix, _ = core.branch_model_matrix(
                policy,
                scenario,
                pair,
                independent,
                horizon,
                models,
                keep_traces=False,
            )
            crn_delta[replicate] = float(
                core.robust_score(crn_matrix[1])
                - core.robust_score(crn_matrix[0])
            )
            independent_delta[replicate] = float(
                core.robust_score(independent_matrix[1])
                - core.robust_score(independent_matrix[0])
            )
    independent_variance = float(np.var(independent_delta, ddof=1))
    crn_variance = float(np.var(crn_delta, ddof=1))
    reduction = 100.0 * (
        independent_variance - crn_variance
    ) / max(independent_variance, 1e-12)
    return {
        "independent_delta": independent_delta,
        "crn_delta": crn_delta,
        "independent_variance": independent_variance,
        "crn_variance": crn_variance,
        "variance_reduction_percent": reduction,
    }


def residual_refinement_probe(
    config: core.ExperimentConfig,
    baseline_path: Path,
    smoke: bool,
) -> dict[str, np.ndarray]:
    baseline = np.load(baseline_path, allow_pickle=True)
    baseline_index = list(baseline["variants"]).index("+ HF calibration")
    trial_count = config.n_trials
    names = np.asarray(["Parametric calibration", "Physics + residual"])
    shape = (trial_count, 2)
    final_steps = np.zeros(shape)
    final_success = np.zeros(shape)
    final_backlog = np.zeros(shape)
    final_rmse = np.zeros(shape)
    correct_yield = np.zeros(shape)
    conditional_accuracy = np.zeros(shape)
    query_counts = np.zeros(shape)
    wall_times = np.zeros(shape)
    rmse_curves = np.zeros((trial_count, 2, config.train_episodes + 1))

    available_baseline_trials = baseline["final_steps"].shape[0]
    if available_baseline_trials < trial_count:
        raise ValueError("Baseline aggregate has fewer paired trials than requested.")

    final_steps[:, 0] = baseline["final_steps"][:trial_count, baseline_index]
    final_success[:, 0] = baseline["final_success"][:trial_count, baseline_index]
    final_backlog[:, 0] = baseline["final_backlog_area"][
        :trial_count, baseline_index
    ]
    final_rmse[:, 0] = baseline["model_rmse"][
        :trial_count, baseline_index, -1
    ]
    correct_yield[:, 0] = baseline["correct_label_yield"][
        :trial_count, baseline_index
    ]
    conditional_accuracy[:, 0] = baseline["conditional_label_accuracy"][
        :trial_count, baseline_index
    ]
    query_counts[:, 0] = np.sum(
        baseline["query_counts"][:trial_count, baseline_index], axis=1
    )
    wall_times[:, 0] = baseline["wall_times"][:trial_count, baseline_index]
    rmse_curves[:, 0] = baseline["model_rmse"][
        :trial_count, baseline_index
    ]

    residual_variant = core.VARIANTS[-1]
    raw_results: list[dict[str, np.ndarray | float]] = []
    for trial in range(trial_count):
        scenario = core.make_validation_scenario(83000 + trial, num_users=4)
        result = core.train_variant(
            residual_variant,
            scenario,
            config,
            93000 + trial,
        )
        result.pop("policy")
        final_steps[trial, 1] = result["final_steps"]
        final_success[trial, 1] = result["final_completed"]
        final_backlog[trial, 1] = result["final_backlog_area"]
        final_rmse[trial, 1] = result["model_rmse"][-1]
        accuracy, yield_value, _ = core.label_statistics(
            result["predicted_labels"], result["audited_labels"]
        )
        correct_yield[trial, 1] = yield_value
        conditional_accuracy[trial, 1] = accuracy
        query_counts[trial, 1] = np.sum(result["query_counts"])
        wall_times[trial, 1] = result["wall_time"]
        rmse_curves[trial, 1] = result["model_rmse"]
        raw_results.append(result)
        print(
            f"residual trial {trial + 1}/{trial_count}: "
            f"steps={final_steps[trial, 1]:.0f}, "
            f"backlog={final_backlog[trial, 1]:.2f}, "
            f"yield={100*yield_value:.1f}%, "
            f"rmse={final_rmse[trial, 1]:.3f}",
            flush=True,
        )

    return {
        "methods": names,
        "final_steps": final_steps,
        "final_success": final_success,
        "final_backlog_area": final_backlog,
        "final_model_rmse": final_rmse,
        "correct_label_yield": correct_yield,
        "conditional_label_accuracy": conditional_accuracy,
        "query_counts": query_counts,
        "wall_times": wall_times,
        "model_rmse_curves": rmse_curves,
        "raw_results": np.asarray(raw_results, dtype=object),
        "smoke": np.asarray(smoke),
    }


def save_crn_outputs(
    result_dir: Path,
    figure_dir: Path,
    table_dir: Path,
    data: dict[str, np.ndarray | float],
    tag: str,
    metadata: dict[str, object],
) -> None:
    np.savez_compressed(
        result_dir / f"crn_variance_probe{tag}.npz",
        **data,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    independent_variance = float(data["independent_variance"])
    crn_variance = float(data["crn_variance"])
    reduction = float(data["variance_reduction_percent"])
    lines = [
        "| Branch comparison | Objective-difference variance $\\downarrow$ |",
        "|---|---:|",
        f"| Independent channel draws | {independent_variance:.3e} |",
        f"| CRN paired draws | **<span style=\"color:#d62728\">"
        f"{crn_variance:.3e}</span>** |",
        f"| **CRN gain** | **{reduction:+.1f}% variance reduction** |",
    ]
    (
        table_dir / f"table_crn_variance_probe{tag}.md"
    ).write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 6))
    samples = [
        np.asarray(data["independent_delta"]),
        np.asarray(data["crn_delta"]),
    ]
    box = ax.boxplot(
        samples,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.8},
    )
    colors = ["#7f7f7f", core.METHOD_COLORS["+ CRN"]]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.30)
    rng = np.random.default_rng(20260730)
    for index, (values, color) in enumerate(zip(samples, colors), start=1):
        ax.scatter(
            rng.normal(index, 0.04, len(values)),
            values,
            s=24,
            color=color,
            alpha=0.65,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Independent\ndraws", "Common random\nnumbers"])
    ax.set_ylabel("Branch objective difference", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"gradient_info_crn_variance{tag}.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def save_residual_outputs(
    result_dir: Path,
    figure_dir: Path,
    table_dir: Path,
    data: dict[str, np.ndarray],
    tag: str,
    metadata: dict[str, object],
) -> None:
    raw_results = data.pop("raw_results")
    np.savez_compressed(
        result_dir / f"residual_calibration_probe{tag}.npz",
        **data,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    raw_dir = result_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for trial, raw in enumerate(raw_results, start=1):
        np.savez_compressed(
            raw_dir / f"residual_calibration_probe{tag}_trial{trial:02d}.npz",
            **raw,
            metadata_json=np.asarray(json.dumps(metadata)),
        )

    med_steps = np.median(data["final_steps"], axis=0)
    med_backlog = np.median(data["final_backlog_area"], axis=0)
    med_rmse = np.median(data["final_model_rmse"], axis=0)
    mean_yield = 100 * np.mean(data["correct_label_yield"], axis=0)
    med_queries = np.median(data["query_counts"], axis=0)
    methods = data["methods"]
    lines = [
        "| Calibrated ensemble | Completion $\\downarrow$ | Backlog $\\downarrow$ | "
        "Correct/branch $\\uparrow$ | WM RMSE $\\downarrow$ | Queries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for index, method in enumerate(methods):
        lines.append(
            f"| {method} | {med_steps[index]:.1f} | "
            f"{med_backlog[index]:.2f} | {mean_yield[index]:.1f}% | "
            f"{med_rmse[index]:.3f} | {med_queries[index]:.1f} |"
        )
    lines.append(
        "| **Residual minus parametric** | "
        f"**{med_steps[0] - med_steps[1]:+.1f} slots** | "
        f"**{med_backlog[0] - med_backlog[1]:+.2f}** | "
        f"**{mean_yield[1] - mean_yield[0]:+.1f} pp** | "
        f"**{med_rmse[0] - med_rmse[1]:+.3f}** | -- |"
    )
    (
        table_dir / f"table_residual_calibration_probe{tag}.md"
    ).write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 6))
    episodes = np.arange(data["model_rmse_curves"].shape[2])
    colors = [
        core.METHOD_COLORS["+ HF calibration"],
        core.METHOD_COLORS["+ Learned residual"],
    ]
    for index, (method, color) in enumerate(zip(methods, colors)):
        median = np.median(data["model_rmse_curves"][:, index], axis=0)
        lower = np.percentile(
            data["model_rmse_curves"][:, index], 25, axis=0
        )
        upper = np.percentile(
            data["model_rmse_curves"][:, index], 75, axis=0
        )
        ax.plot(episodes, median, color=color, linewidth=2.2, label=method)
        ax.fill_between(
            episodes, lower, upper, color=color, alpha=0.14, linewidth=0
        )
    ax.set_xlabel("Calibration update", fontsize=14)
    ax.set_ylabel("Held-out WM rate RMSE", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"gradient_info_residual_calibration{tag}.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def write_refined_report(
    path: Path,
    config: core.ExperimentConfig,
    combined: dict[str, np.ndarray],
    probe_result_dir: Path,
) -> None:
    med_steps = np.median(combined["final_steps"], axis=0)
    success_rate = np.mean(combined["final_success"], axis=0)
    med_backlog = np.median(combined["final_backlog_area"], axis=0)
    mean_yield = np.mean(combined["correct_label_yield"], axis=0)
    med_rmse = np.median(combined["model_rmse"][:, :, -1], axis=0)
    med_initial_rmse = np.median(combined["model_rmse"][:, :, 0], axis=0)
    queries = np.sum(combined["query_counts"], axis=(0, 2))
    gradient_logdet = np.mean(combined["gradient_logdet"], axis=(0, 2))
    crn_file = np.load(
        probe_result_dir / "crn_variance_probe_validate.npz",
        allow_pickle=True,
    )
    independent_variance = float(crn_file["independent_variance"])
    crn_variance = float(crn_file["crn_variance"])
    crn_reduction = float(crn_file["variance_reduction_percent"])

    lines = [
        "# Gradient-Information World-Model Exploration",
        "",
        "## Controlled protocol",
        "",
        f"- {config.n_trials} paired corridor layouts and "
        f"{config.train_episodes} policy updates per variant.",
        f"- {config.pool_branches} candidate branches are generated and "
        f"{config.selected_branches} are retained per update.",
        "- Every incremental variant uses the same policy architecture, "
        "initialization, optimizer, update count, and high-fidelity evaluation model.",
        "- The high-fidelity model contains anisotropic attenuation outside the "
        "radial parametric ensemble family.",
        "- Only the final two variants receive high-fidelity training queries, "
        "with at most one queried branch per update.",
        "- The learned-residual variant uses four calibration warm-up episodes "
        "because an uncalibrated residual cannot use its own confidence as a "
        "reliable query gate.",
        "",
        "## Incremental results",
        "",
        "| Variant | Completion | Success | Backlog | Correct/branch | WM RMSE | Queries |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, variant in enumerate(core.VARIANTS):
        lines.append(
            f"| {variant.name} | {med_steps[index]:.1f} | "
            f"{100*success_rate[index]:.1f}% | {med_backlog[index]:.2f} | "
            f"{100*mean_yield[index]:.1f}% | {med_rmse[index]:.3f} | "
            f"{queries[index]:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism checks",
            "",
            "Gradient-information selection increased the confidence-weighted "
            f"gradient log-determinant from {gradient_logdet[0]:.3f} to "
            f"{gradient_logdet[1]:.3f}. Relative to action-space diversity, it "
            f"reduced median completion time by {med_steps[0]-med_steps[1]:.1f} "
            f"slots, reduced backlog area by {med_backlog[0]-med_backlog[1]:.2f}, "
            f"increased success from {100*success_rate[0]:.0f}% to "
            f"{100*success_rate[1]:.0f}%, and increased correct accepted labels "
            f"per audited branch from {100*mean_yield[0]:.1f}% to "
            f"{100*mean_yield[1]:.1f}%.",
            "",
            "Confidence-weighted local positive distillation and local negative "
            f"rejection provided a further {med_steps[1]-med_steps[2]:.1f}-slot "
            f"completion reduction and a {med_backlog[1]-med_backlog[2]:.2f} "
            "backlog reduction. The state-dependent trust region then reduced "
            f"completion by another {med_steps[2]-med_steps[3]:.1f} slot and "
            f"backlog by {med_backlog[2]-med_backlog[3]:.2f}. These are "
            "directional results from two paired layouts, not statistical evidence.",
            "",
            "The direct CRN probe used 32 repeated comparisons of the same action "
            f"branch. Pairing channel draws reduced objective-difference variance "
            f"from {independent_variance:.3e} to {crn_variance:.3e}, a "
            f"{crn_reduction:.4f}% reduction. In the incremental policy run, CRN "
            "also made the fixed model ensemble abstain on more branches. Its low "
            "label yield therefore reflects exposed systematic model bias rather "
            "than stochastic comparison noise. Calibration partially recovered "
            "label coverage.",
            "",
            "Parametric high-fidelity calibration reduced median held-out rate "
            f"RMSE from {med_initial_rmse[-2]:.3f} to {med_rmse[-2]:.3f}. The "
            "physics-plus-RBF-residual ensemble reduced its own RMSE from "
            f"{med_initial_rmse[-1]:.3f} to {med_rmse[-1]:.3f} and increased "
            f"correct label yield from {100*mean_yield[-2]:.1f}% to "
            f"{100*mean_yield[-1]:.1f}% relative to parametric calibration. It "
            "did not outperform the parametric model in absolute RMSE or "
            "end-to-end mission metrics under this query budget.",
            "",
            "## Supported conclusions",
            "",
            "The small-scale experiment supports gradient-information branch "
            "selection, confidence-weighted local updates, state-dependent trust "
            "regions, and CRN as distinct mechanisms. It also validates that "
            "limited high-fidelity feedback can calibrate a misspecified "
            "world-model ensemble.",
            "",
            "## Unsupported conclusions",
            "",
            "The experiment does not establish statistical significance, universal "
            "task improvement, or superiority of learned residuals over "
            "low-dimensional parametric calibration. The learned residual is "
            "currently a more flexible but more data-hungry replacement whose "
            "decision-relevant labels improve before its global rate prediction "
            "becomes competitive.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_incremental_results(
    root: Path,
    baseline_path: Path,
    probe_result_dir: Path,
    config: core.ExperimentConfig,
    tag: str,
) -> Path:
    baseline_file = np.load(baseline_path, allow_pickle=True)
    combined = {key: baseline_file[key].copy() for key in baseline_file.files}
    residual_index = len(core.VARIANTS) - 1
    raw_results = []
    for trial in range(config.n_trials):
        raw_path = (
            probe_result_dir
            / "raw"
            / f"residual_calibration_probe{tag}_trial{trial + 1:02d}.npz"
        )
        raw_file = np.load(raw_path, allow_pickle=True)
        raw_results.append(
            {
                key: raw_file[key]
                for key in raw_file.files
                if key != "metadata_json"
            }
        )

    scalar_map = {
        "final_steps": "final_steps",
        "final_success": "final_completed",
        "final_backlog_area": "final_backlog_area",
        "final_remaining": "final_remaining",
        "wall_times": "wall_time",
        "query_buffer_sizes": "query_buffer_size",
    }
    episode_map = {
        "training_objectives": "training_objective",
        "nominal_objectives": "nominal_objective",
        "update_norms": "update_norm",
        "gradient_logdet": "gradient_logdet",
        "action_logdet": "action_logdet",
        "positive_counts": "positive_counts",
        "negative_counts": "negative_counts",
        "ambiguous_counts": "ambiguous_counts",
        "query_counts": "query_counts",
        "model_rmse": "model_rmse",
        "predicted_labels": "predicted_labels",
        "audited_labels": "audited_labels",
        "convergence_steps": "checkpoint_steps",
        "convergence_backlog_area": "checkpoint_backlog",
    }
    for trial, result in enumerate(raw_results):
        for aggregate_key, raw_key in scalar_map.items():
            combined[aggregate_key][trial, residual_index] = result[raw_key]
        for aggregate_key, raw_key in episode_map.items():
            combined[aggregate_key][trial, residual_index] = result[raw_key]
        combined["trajectories"][trial, residual_index] = result["trajectory"]
        combined["policy_vectors"][trial, residual_index] = result[
            "policy_vector"
        ]
        accuracy, yield_value, accepted = core.label_statistics(
            result["predicted_labels"], result["audited_labels"]
        )
        combined["conditional_label_accuracy"][
            trial, residual_index
        ] = accuracy
        combined["correct_label_yield"][trial, residual_index] = yield_value
        combined["accepted_label_counts"][trial, residual_index] = accepted

    metadata = json.loads(str(combined["metadata_json"]))
    metadata["residual_refinement"] = {
        "model": "physics core plus smooth bootstrapped RBF residual",
        "maximum_query_steps": core.MAX_QUERY_STEPS,
        "minimum_calibration_warmup_episodes": 4,
        "source": str(
            probe_result_dir / f"residual_calibration_probe{tag}.npz"
        ),
    }
    combined["metadata_json"] = np.asarray(json.dumps(metadata))
    output_path = (
        baseline_path.parent
        / "gradient_information_wm_validation_refined.npz"
    )
    np.savez_compressed(output_path, **combined)

    refined_raw_dir = baseline_path.parent / "refined_raw"
    refined_raw_dir.mkdir(parents=True, exist_ok=True)
    residual_token = "plus_learned_residual"
    for trial, result in enumerate(raw_results):
        baseline_raw_path = (
            baseline_path.parent
            / "raw"
            / f"gradient_information_wm_validate_trial{trial + 1:02d}.npz"
        )
        baseline_raw = np.load(baseline_raw_path, allow_pickle=True)
        refined_payload = {
            key: baseline_raw[key]
            for key in baseline_raw.files
        }
        for key, value in result.items():
            refined_payload[f"{residual_token}_{key}"] = np.asarray(value)
        refined_payload["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(
            refined_raw_dir
            / f"gradient_information_wm_refined_trial{trial + 1:02d}.npz",
            **refined_payload,
        )

    figure_dir = root / "paper_assets" / "figures"
    table_dir = root / "paper_assets" / "tables"
    core.save_table(
        table_dir / "table_gradient_information_wm_refined.md",
        combined["final_steps"],
        combined["final_success"],
        combined["final_backlog_area"],
        combined["correct_label_yield"],
        combined["model_rmse"][:, :, -1],
        combined["wall_times"],
    )
    core.plot_results(
        figure_dir,
        combined["final_steps"],
        combined["final_backlog_area"],
        combined["correct_label_yield"],
        combined["gradient_logdet"],
        combined["model_rmse"],
    )
    write_refined_report(
        baseline_path.parent / "validation_report_refined.md",
        config,
        combined,
        probe_result_dir,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    result_dir = root / "results_extended" / "gradient_information_wm" / "probes"
    figure_dir = root / "paper_assets" / "figures"
    table_dir = root / "paper_assets" / "tables"
    for directory in (result_dir, figure_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        replicates = 4
        config = core.ExperimentConfig(1, 1, 10, 30, 3, 2, 1, "_smoke")
        baseline_path = (
            root
            / "results_extended"
            / "gradient_information_wm"
            / "gradient_information_wm_validation_smoke.npz"
        )
        tag = "_smoke"
    else:
        replicates = 32
        config = core.ExperimentConfig(2, 6, 25, 130, 5, 3, 2, "_validate")
        baseline_path = (
            root
            / "results_extended"
            / "gradient_information_wm"
            / "gradient_information_wm_validation_validate.npz"
        )
        tag = "_validate"

    if args.finalize_only:
        output_path = finalize_incremental_results(
            root, baseline_path, result_dir, config, tag
        )
        print(f"Saved {output_path}")
        return

    start = time.perf_counter()
    crn_data = crn_variance_probe(
        replicates=replicates,
        horizon=config.train_horizon,
        scenario_seed=83000,
        policy_seed=93000,
    )
    metadata = {
        "replicates": replicates,
        "config": asdict(config),
        "device": str(core.DEVICE),
        "baseline_path": str(baseline_path),
    }
    save_crn_outputs(
        result_dir, figure_dir, table_dir, crn_data, tag, metadata
    )
    residual_data = residual_refinement_probe(
        config, baseline_path, args.smoke
    )
    save_residual_outputs(
        result_dir, figure_dir, table_dir, residual_data, tag, metadata
    )
    output_path = finalize_incremental_results(
        root, baseline_path, result_dir, config, tag
    )
    print(
        f"CRN variance reduction: "
        f"{float(crn_data['variance_reduction_percent']):.1f}%"
    )
    print(f"Probe wall time: {time.perf_counter() - start:.2f}s")
    print(f"Saved {output_path}")
    if args.smoke:
        print("Smoke test PASSED")


if __name__ == "__main__":
    main()

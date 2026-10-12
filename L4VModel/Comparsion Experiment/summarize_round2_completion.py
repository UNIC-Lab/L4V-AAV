"""Rebuild the verified round-2 summary from persisted formal artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_round2"


def csi_pooled_summary():
    path = RESULTS / "csi_adaptation" / "csi_adaptation.npz"
    with np.load(path, allow_pickle=True) as raw:
        trials = int(raw["trials"])
        methods = {}
        for method in ("backlog_only", "current_csi"):
            keys = {
                "success": "success",
                "completion": "completion_slots",
                "average_completion": "average_completion_slots",
                "correlation": "allocation_log_csi_correlation",
                "best_share": "best_csi_user_allocation_share",
                "sensitivity": "allocation_permutation_sensitivity",
                "runtime": "direct_inference_ms",
            }
            values = {
                name: np.concatenate(
                    [
                        raw[f"{method}_trial{trial:02d}_{key}"]
                        for trial in range(1, trials + 1)
                    ]
                )
                for name, key in keys.items()
            }
            methods[method] = {
                "rollouts": int(values["success"].size),
                "completion_fraction": float(np.mean(values["success"])),
                "median_completion_slots": float(np.median(values["completion"])),
                "median_average_completion_slots": float(
                    np.median(values["average_completion"])
                ),
                "median_allocation_log_csi_correlation": float(
                    np.median(values["correlation"])
                ),
                "median_best_csi_user_allocation_share": float(
                    np.median(values["best_share"])
                ),
                "median_allocation_permutation_sensitivity": float(
                    np.median(values["sensitivity"])
                ),
                "median_direct_inference_ms": float(
                    np.median(values["runtime"])
                ),
            }
        return {
            "protocol": "round2_csi_adaptation_formal",
            "paired_layouts": trials,
            "evaluation_rollouts_per_layout": int(raw["eval_rollouts"]),
            "updates_per_policy": int(raw["updates"]),
            "shadow_std": float(raw["shadow_std"]),
            "rician_k": float(raw["rician_k"]),
            "methods": methods,
        }


def main():
    csi = csi_pooled_summary()
    csi_path = RESULTS / "csi_adaptation" / "summary_pooled.json"
    csi_path.write_text(json.dumps(csi, indent=2), encoding="utf-8")
    paired = json.loads(
        (RESULTS / "qcqp" / "l4v_paired_summary.json").read_text(encoding="utf-8")
    )
    verified_path = RESULTS / "round2_verified_summary.json"
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    verified["qcqp_paired_l4v"] = paired
    verified["csi_adaptation"] = csi
    verified_path.write_text(json.dumps(verified, indent=2), encoding="utf-8")
    print(json.dumps({"qcqp_paired_l4v": paired, "csi_adaptation": csi}, indent=2))


if __name__ == "__main__":
    main()

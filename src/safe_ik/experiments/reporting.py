"""Serialization of completed experiment metrics."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .analysis import _paired_analysis
from .models import METHODS


def _json_default(value):
    """Convert NumPy scalar diagnostics to JSON-native scalar values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_summary(output_dir: Path, records, experiment, config):
    analysis = _paired_analysis(records)
    payload = {
        "settings": {
            **asdict(experiment),
            "resolved_control_mode": config.control_mode,
            "kinematic": config.simulation.kinematic,
            "link_lengths": list(config.safety.link_lengths),
        },
        "cases": records,
        "analysis": analysis,
        "sampling": {
            "ik_rejections_before_acceptance": sum(
                record.get("ik_rejections_before_acceptance", 0) for record in records
            ),
            "note": "Only cases with at least one collision-free IK branch are executed in both methods.",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, default=_json_default), encoding="utf-8"
    )
    scalar_fields = (
        "success",
        "position_error",
        "yaw_error",
        "min_barrier",
        "active_link",
        "failure_reason",
        "completion_time",
        "control_effort",
        "max_joint_velocity",
        "mean_clf_slack",
        "max_clf_slack",
        "nonzero_clf_slack_ratio_to_goal",
        "total_qp_solve_time",
        "qp_infeasible_count",
        "qp_solver_failure_count",
    )
    columns = ["case_id", "target_x", "target_y", "target_yaw"] + [
        f"{method}_{field}" for method in METHODS for field in scalar_fields
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {
                "case_id": record["case_id"],
                "target_x": record["target"][0],
                "target_y": record["target"][1],
                "target_yaw": record["target_yaw"],
            }
            for method in METHODS:
                row.update(
                    {
                        f"{method}_{field}": record[method][field]
                        for field in scalar_fields
                    }
                )
            writer.writerow(row)

"""Pure trajectory metrics and paired experiment analysis."""

import numpy as np

from .models import COMPARISONS, METHODS, OUTCOME_LABELS


def classify_outcome(nonsequential_success: bool, sequential_success: bool) -> str:
    """Return one of the four comparison categories in a stable order."""
    return OUTCOME_LABELS[(bool(nonsequential_success), bool(sequential_success))]


def _sample_interval(times: np.ndarray) -> float:
    """Return the recorder's nominal simulation interval."""
    if len(times) < 2:
        return 0.0
    return float(np.median(np.diff(times)))


def _nonzero_slack_ratio_to_goal(
    slack, goal_reached, active_threshold: float
) -> float | None:
    """Return active-slack samples divided by control samples to first goal hit."""
    slack = np.asarray(slack, dtype=float)
    reached_indices = np.flatnonzero(goal_reached)
    if not slack.size or not len(reached_indices):
        return None
    # The goal-reaching state is included: it has one control/QP sample at
    # the same simulation timestep, matching the recorded control period.
    until_goal = slack[: reached_indices[0] + 1]
    finite = until_goal[np.isfinite(until_goal)]
    if not finite.size:
        return None
    return float(np.mean(finite > active_threshold))


def _stage_durations(
    active_link: np.ndarray, dt: float, dof: int, sequential: bool
) -> tuple[float, ...]:
    """Measure time spent with each sequential task endpoint active."""
    if not sequential or active_link.size == 0:
        return ()
    return tuple(
        float(np.count_nonzero(active_link == index) * dt)
        for index in range(1, dof + 1)
    )


def _failure_reason(
    success: bool,
    min_barrier: float,
    barrier_tolerance: float,
    qp_infeasible_count: int,
    sequential: bool,
    active_link: int,
    dof: int,
    switch_reasons: tuple[str, ...],
    qp_solver_failure_count: int = 0,
) -> str | None:
    """Classify an unsuccessful rollout without conflating it with paired outcome."""
    if success:
        return None
    if min_barrier < barrier_tolerance:
        return "safety_violation"
    if qp_infeasible_count:
        return "qp_infeasible"
    if qp_solver_failure_count:
        return "qp_solver_failure"
    if (
        sequential
        and active_link < dof
        and any("CBF" in reason for reason in switch_reasons)
    ):
        return "cbf_boundary_stall"
    return "timeout"


def _distribution(values):
    """Summarize finite scalar values in a JSON-friendly form."""
    values = np.asarray([value for value in values if value is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _paired_analysis(records):
    """Return outcome, failure, and paired metric summaries for one run."""
    result = {
        "case_count": len(records),
        "methods": {},
        "comparisons": {},
    }
    metrics = (
        "position_error",
        "yaw_error",
        "min_barrier",
        "control_effort",
        "max_joint_velocity",
        "total_qp_solve_time",
        "mean_qp_solve_time",
        "mean_clf_slack",
        "max_clf_slack",
        "nonzero_clf_slack_ratio_to_goal",
    )
    for method in METHODS:
        runs = [record[method] for record in records]
        failure_counts = {}
        for run in runs:
            if run["failure_reason"] is not None:
                failure_counts[run["failure_reason"]] = (
                    failure_counts.get(run["failure_reason"], 0) + 1
                )
        result["methods"][method] = {
            "success_count": sum(run["success"] for run in runs),
            "success_rate": float(np.mean([run["success"] for run in runs])),
            "failure_reasons": failure_counts,
            "metrics": {
                metric: _distribution([run[metric] for run in runs])
                for metric in metrics
            },
        }
    for name, (baseline, candidate) in COMPARISONS.items():
        outcomes = {label: 0 for label in OUTCOME_LABELS.values()}
        for record in records:
            outcomes[record["comparisons"][name]] += 1
        successful_pairs = [
            record
            for record in records
            if record[baseline]["success"] and record[candidate]["success"]
        ]
        deltas = {}
        for metric in (
            "completion_time",
            "control_effort",
            "min_barrier",
            "max_clf_slack",
            "nonzero_clf_slack_ratio_to_goal",
            "position_error",
            "yaw_error",
        ):
            values = [
                record[candidate][metric] - record[baseline][metric]
                for record in successful_pairs
                if record[candidate][metric] is not None
                and record[baseline][metric] is not None
            ]
            deltas[f"{candidate}_minus_{baseline}_{metric}"] = _distribution(values)
        result["comparisons"][name] = {
            "baseline": baseline,
            "candidate": candidate,
            "outcomes": outcomes,
            "paired_both_success": deltas,
        }
    return result

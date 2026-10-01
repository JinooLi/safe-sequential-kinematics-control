"""Shared, versioned distance-accuracy policy for offline Panda RDF evaluation."""

from numbers import Real

import numpy as np

ACCURACY_DOMAIN = "mesh_exterior_including_surface_v1"
NEAR_DISTANCE = 0.060
ASYMMETRIC_POLICY = "asymmetric_exterior_near_error_v1"


def validation_policy(options):
    """Accuracy policy is deliberately separate from geometry/training settings."""
    return {
        "accuracy_policy": (
            ASYMMETRIC_POLICY
            if options.underestimation_limit is not None
            else "symmetric_absolute_error_v1"
        ),
        "accuracy_domain": ACCURACY_DOMAIN,
        "near_distance": NEAR_DISTANCE,
        "overestimation_limit": options.error_margin,
        "underestimation_limit": (
            options.error_margin
            if options.underestimation_limit is None
            else options.underestimation_limit
        ),
    }


def _positive_limit(value):
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and np.isfinite(value)
        and value > 0
    )


def accuracy_masks(truth):
    """Use the reference sign, never the prediction, to select evaluated points."""
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 1 or not len(truth) or not np.isfinite(truth).all():
        raise ValueError("reference distances must be a nonempty finite vector")
    exterior = truth >= 0.0
    return exterior, exterior & (truth <= NEAR_DISTANCE)


def distance_accuracy_report(
    points,
    truth,
    predicted,
    error_margin,
    *,
    global_count=3000,
    underestimation_limit=None,
):
    """Accuracy statistics exclude mesh interiors; empty near coverage fails closed.

    All arrays still contain the original validation samples. Excluding interiors
    does not permit nonfinite model output or alter the signed training targets.
    """
    points = np.asarray(points, dtype=float)
    truth, predicted = np.asarray(truth, dtype=float), np.asarray(
        predicted, dtype=float
    )
    exterior, near = accuracy_masks(truth)
    if (
        points.shape != (len(truth), 3)
        or predicted.shape != truth.shape
        or not np.isfinite(points).all()
        or not np.isfinite(predicted).all()
        or not _positive_limit(error_margin)
        or (
            underestimation_limit is not None
            and not _positive_limit(underestimation_limit)
        )
        or type(global_count) is not int
        or not 0 <= global_count <= len(truth)
    ):
        raise ValueError("invalid distance accuracy inputs")
    if not near.any():
        raise ValueError("no exterior near-surface validation points; cannot approve")
    error = np.abs(predicted - truth)
    if not np.isfinite(error[exterior]).all():
        raise ValueError("nonfinite exterior distance error")
    global_exterior = exterior.copy()
    global_exterior[global_count:] = False
    worst = np.flatnonzero(near)[np.argsort(-error[near], kind="stable")[:5]]
    near_max = float(error[near].max())
    report = {
        "accuracy_domain": ACCURACY_DOMAIN,
        "near_distance": NEAR_DISTANCE,
        "sample_count": len(truth),
        "evaluated_count": int(exterior.sum()),
        "interior_excluded_count": int((~exterior).sum()),
        "global_evaluated_count": int(global_exterior.sum()),
        "near_count": int(near.sum()),
        "near_max_error": near_max,
        "near_mean_error": float(error[near].mean()),
        "near_p95_error": float(np.quantile(error[near], 0.95)),
        "global_max_error": (
            float(error[global_exterior].max()) if global_exterior.any() else None
        ),
        "mean_error": float(error[exterior].mean()),
        "sign_errors": int(np.sum(predicted[exterior] < 0)),
        "distance_passed": bool(near_max <= error_margin),
        "worst_near_points": [
            {
                "point": points[i].tolist(),
                "truth": float(truth[i]),
                "prediction": float(predicted[i]),
                "absolute_error": float(error[i]),
            }
            for i in worst
        ],
    }
    if underestimation_limit is not None:
        signed = predicted - truth
        over = float(np.maximum(signed[near], 0).max())
        under = float(np.maximum(-signed[near], 0).max())
        report.update(
            accuracy_policy=ASYMMETRIC_POLICY,
            overestimation_limit=float(error_margin),
            underestimation_limit=float(underestimation_limit),
            near_max_overestimate=over,
            near_max_underestimate=under,
            near_overestimation_exceedance_count=int(
                np.sum(signed[near] > error_margin)
            ),
            near_underestimation_exceedance_count=int(
                np.sum(signed[near] < -underestimation_limit)
            ),
            near_absolute_above_10mm_count=int(np.sum(error[near] > 0.010)),
            distance_passed=bool(
                over <= error_margin and under <= underestimation_limit
            ),
        )
    return report


def require_distance_policy(report, options):
    """Fail closed on old/partial/direction-swapped reports under the new policy."""
    under = options.underestimation_limit
    if under is None:
        if report.get("accuracy_policy") not in (None, "symmetric_absolute_error_v1"):
            raise ValueError("accuracy policy mismatch; explicit revalidation required")
        if report["near_max_error"] > options.error_margin:
            raise ValueError("symmetric distance validation failed")
        return
    expected = validation_policy(options)
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("accuracy policy mismatch; explicit revalidation required")
    maxima = [
        report.get(k)
        for k in ("near_max_underestimate", "near_max_overestimate", "near_max_error")
    ]
    if any(
        isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v < 0
        for v in maxima
    ):
        raise ValueError("invalid signed distance maxima")
    u, o, absolute = maxima
    if absolute != max(u, o) or u > under or o > options.error_margin:
        raise ValueError("asymmetric distance validation failed")
    for key in (
        "near_underestimation_exceedance_count",
        "near_overestimation_exceedance_count",
    ):
        if type(report.get(key)) is not int or report[key] != 0:
            raise ValueError("asymmetric exceedance counts fail validation")

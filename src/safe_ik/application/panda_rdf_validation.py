"""Shared, versioned distance-accuracy policy for offline Panda RDF evaluation."""

import numpy as np

ACCURACY_DOMAIN = "mesh_exterior_including_surface_v1"
NEAR_DISTANCE = 0.060


def accuracy_masks(truth):
    """Use the reference sign, never the prediction, to select evaluated points."""
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 1 or not len(truth) or not np.isfinite(truth).all():
        raise ValueError("reference distances must be a nonempty finite vector")
    exterior = truth >= 0.0
    return exterior, exterior & (truth <= NEAR_DISTANCE)


def distance_accuracy_report(
    points, truth, predicted, error_margin, *, global_count=3000
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
        or not np.isfinite(error_margin)
        or error_margin <= 0
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
    return {
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

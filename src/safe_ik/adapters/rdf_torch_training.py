"""Optional PyTorch implementation of the existing float64 recursive ridge fit.

Only offline training uses this module. No clipping, gradient normalization,
precision reduction or implicit CPU fallback is introduced. The default keeps
the original objective; weighted offline experiments require explicit opt-in.
"""

from math import comb

import numpy as np


def torch_training_environment(device="cuda"):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch required; see configs/panda_rdf_gpu_requirements.txt"
        ) from exc
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("training device must be cpu or cuda")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    return {
        "backend": "pytorch",
        "device": str(selected),
        "dtype": "float64",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(selected) if selected.type == "cuda" else None
        ),
    }


def recursive_ridge_torch(
    model,
    batches,
    regularization=1e-4,
    *,
    device="cuda",
    progress=None,
    resume=None,
    checkpoint=None,
    checkpoint_interval=400,
    weighted=False,
    allow_weighted_resume=False,
):
    """Same normalization, Bernstein basis and Algorithm 1 as the NumPy path.

    A CPU torch device is allowed explicitly for parity tests, not as fallback.
    Batch sampling remains with the caller so CPU and CUDA consume identical data.
    On resume, batches must contain ONLY the unprocessed suffix. Progress and
    checkpoint counters are cumulative. Checkpoints receive normalized weights.

    Explicit weighted mode consumes (points, targets, positive sample weights).
    Both features and normalized targets are multiplied by sqrt(weight), giving
    sum_i weight_i * residual_i**2 + regularization * ||w||**2. It must start
    from scratch by default. A verified weighted parent can be continued only
    with allow_weighted_resume=True; callers must verify and preserve its weight
    arrays/prefix and record the appended data lineage in the new checkpoint.
    """
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be positive")
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer")
    if type(weighted) is not bool:
        raise ValueError("weighted must be boolean")
    if type(allow_weighted_resume) is not bool:
        raise ValueError("allow_weighted_resume must be boolean")
    if allow_weighted_resume and (resume is None or not weighted):
        raise ValueError("weighted resume opt-in requires a weighted parent")
    if resume is not None:
        objective = resume.metadata.get("objective", "unweighted")
        if weighted or objective != "unweighted":
            if not (allow_weighted_resume and weighted and objective == "weighted"):
                raise ValueError(
                    "weighted experiments cannot resume checkpoints without explicit opt-in"
                )
            context = resume.metadata.get("context", {})
            prefix = resume.metadata.get("sample_weight_prefix_sha256", {})
            digest = context.get("sample_weights_sha256", "")
            if (
                context.get("objective") != "weighted"
                or not prefix
                or len(digest) != 64
                or any(len(v) != 64 for v in prefix.values())
            ):
                raise ValueError("weighted resume requires weight provenance")
        resume.validate(model, regularization)
    torch_training_environment(device)
    import torch

    dtype = torch.float64
    selected = torch.device(device)
    size, count = model.weights.size, model.count
    with torch.no_grad():
        if resume is None:
            covariance = torch.eye(size, dtype=dtype, device=selected) / regularization
            weights = torch.zeros(size, dtype=dtype, device=selected)
            completed, seen = 0, 0
        else:
            # tensor() copies read-only mmap arrays; training cannot mutate snapshots.
            covariance = torch.tensor(resume.covariance, dtype=dtype, device=selected)
            weights = torch.tensor(resume.weights, dtype=dtype, device=selected)
            completed = resume.metadata["completed_steps"]
            seen = resume.metadata["consumed_samples"]
        powers = torch.arange(count, dtype=dtype, device=selected)
        coefficients = torch.tensor(
            [comb(count - 1, i) for i in range(count)], dtype=dtype, device=selected
        )
        center = torch.as_tensor(model.center, dtype=dtype, device=selected)
        step = completed
        last_saved = None
        for step, batch in enumerate(batches, start=completed + 1):
            if weighted:
                points, targets, sample_weights = batch
            else:
                points, targets = batch
            points, targets = np.asarray(points, dtype=float), np.asarray(
                targets, dtype=float
            )
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or not len(points)
                or targets.shape != (len(points),)
                or not np.isfinite(points).all()
                or not np.isfinite(targets).all()
                or np.any(
                    np.linalg.norm(points - model.center, axis=1) > model.radius + 1e-12
                )
            ):
                raise ValueError(
                    "invalid training batch or points outside training sphere"
                )
            t = (torch.as_tensor(points, dtype=dtype, device=selected) - center) / (
                2 * model.radius
            ) + 0.5
            bases = [
                coefficients
                * t[:, axis, None] ** powers
                * (1 - t[:, axis, None]) ** (count - 1 - powers)
                for axis in range(3)
            ]
            phi = torch.einsum("bi,bj,bk->bijk", *bases).reshape(len(points), size)
            root_weights = None
            if weighted:
                sample_weights = np.asarray(sample_weights, dtype=float)
                if (
                    sample_weights.shape != (len(points),)
                    or not np.isfinite(sample_weights).all()
                    or np.any(sample_weights <= 0)
                ):
                    raise ValueError("sample weights must be finite and positive")
                root_weights = torch.as_tensor(
                    np.sqrt(sample_weights), dtype=dtype, device=selected
                )
                phi *= root_weights[:, None]
            cross = covariance @ phi.T
            system = torch.eye(len(points), dtype=dtype, device=selected) + phi @ cross
            # Match scipy.linalg.solve(assume_a='pos')'s upper triangle convention.
            factor = torch.linalg.cholesky(system, upper=True)
            gain = torch.cholesky_solve(cross.T, factor, upper=True).T
            targets_t = (
                torch.as_tensor(targets, dtype=dtype, device=selected) / model.radius
            )
            if root_weights is not None:
                targets_t = targets_t * root_weights
            weights += gain @ (targets_t - phi @ weights)
            covariance -= gain @ cross.T
            covariance = (covariance + covariance.T) * 0.5
            if (
                not torch.isfinite(weights).all().item()
                or not torch.isfinite(covariance).all().item()
            ):
                raise ValueError("nonfinite recursive regression state")
            seen += len(points)
            if checkpoint is not None and step % checkpoint_interval == 0:
                checkpoint(step, seen, weights, covariance)
                last_saved = step
            if progress is not None:
                progress(step)
        if step == completed:
            raise ValueError("training batches must not be empty")
        if checkpoint is not None and last_saved != step:
            checkpoint(step, seen, weights, covariance)
        model.weights = (
            (weights * model.radius).cpu().numpy().reshape(model.weights.shape)
        )
    return model

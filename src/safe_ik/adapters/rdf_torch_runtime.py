"""Optional batched float64 RDF inference; never trains or changes artifacts.

NumPy/Pinocchio remain the reference and the source of frame poses. Coefficients
and surface queries stay on the selected device. Only active pairs enter this
pipeline; top-k, spatial gradients, relative motion and soft-min stay on-device.
"""

from math import comb

import numpy as np


class TorchRDFPairs:
    def __init__(self, models, queries, pairs, options, *, device="cuda"):
        try:
            import torch
        except ImportError as exc:
            raise ValueError(
                "CUDA RDF requires PyTorch: use uv run --extra cuda"
            ) from exc
        selected = torch.device(device)
        if selected.type not in ("cpu", "cuda"):
            raise ValueError("RDF tensor device must be cpu or cuda")
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA RDF requested but CUDA is unavailable; no CPU fallback"
            )
        self.torch, self.device = torch, selected
        self.names = tuple(models)
        self.pairs, self.options = tuple(pairs), options
        counts = {m.rdf.count for m in models.values()}
        if len(counts) != 1:
            raise ValueError(
                "batched CUDA RDF requires one shared Bernstein basis count"
            )
        self.count = counts.pop()
        index = {n: i for i, n in enumerate(self.names)}

        def tensor(value, dtype=torch.float64):
            return torch.tensor(value, dtype=dtype, device=selected)

        self.source = tensor([index[a] for a, _ in pairs], torch.long)
        self.target = tensor([index[b] for _, b in pairs], torch.long)
        self.weights = tensor(np.stack([models[n].rdf.weights for n in self.names]))
        self.centers = tensor(np.stack([models[n].rdf.center for n in self.names]))
        self.radii = tensor([models[n].rdf.radius for n in self.names])
        self.inner = tensor([models[n].inner_radius for n in self.names])
        self.maximum = tensor([models[n].surface_maximum for n in self.names])
        self.sizes = np.array([len(queries[a]) for a, _ in pairs], dtype=int)
        if not len(self.sizes) or np.any(self.sizes < 1):
            raise ValueError("RDF pairs require nonempty source queries")
        size = int(self.sizes.max())
        points = np.zeros((len(pairs), size, 3))
        for i, (source, _) in enumerate(pairs):
            points[i, : self.sizes[i]] = queries[source]
        if not np.isfinite(points).all():
            raise ValueError("RDF queries must be finite")
        self.points = tensor(points)
        self.lengths = tensor(self.sizes, torch.long)
        self.positions = torch.arange(size, device=selected)
        self.powers = [tensor(np.arange(n)) for n in (self.count, self.count - 1)]
        self.binomials = [
            tensor([comb(n - 1, i) for i in range(n)])
            for n in (self.count, self.count - 1)
        ]
        self.metadata = {
            "backend": "pytorch_batched",
            "device": str(selected),
            "dtype": "float64",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(selected)
                if selected.type == "cuda"
                else None
            ),
            "timing_layout": "distance=whole tensor batch wall time incl. transfers, selection and derivatives; gradient=CPU gate/blend; selection=0",
        }
        self.last_active_pairs = 0

    def _basis(self, t, low=False):
        powers, coefficients = self.powers[low], self.binomials[low]
        return (
            coefficients
            * t[..., None] ** powers
            * (1 - t[..., None]) ** (len(powers) - 1 - powers)
        )

    def _contract(self, weights, bases):
        # Sequential contractions avoid an N x 24^3 feature tensor.
        x, y, z = bases
        size = weights.shape[-1]
        yz = (x @ weights.reshape(len(weights), size, -1)).reshape(
            *x.shape[:2], size, size
        )
        return ((yz * y[..., :, None]).sum(-2) * z).sum(-1)

    def _field(self, points, targets, gradient=False):
        torch = self.torch
        center = self.centers[targets, None]
        radius, inner = self.radii[targets, None], self.inner[targets, None]
        offset = points - center
        rho = torch.linalg.vector_norm(offset, dim=-1)
        outer = rho >= radius
        # Fixed padded batches avoid host synchronization for dynamic masks.
        # Outside r1 evaluate a harmless center placeholder, NEVER extrapolate
        # the polynomial; only the analytic extension contributes there.
        t = (
            torch.where(outer[..., None], torch.zeros_like(offset), offset)
            / (2 * radius[..., None])
            + 0.5
        )
        bases = [self._basis(t[..., axis]) for axis in range(3)]
        coefficients = self.weights[targets]
        f = self._contract(coefficients, bases)
        F = self.maximum[targets, None] + rho - inner
        u = ((rho - inner) / (radius - inner)).clamp(0, 1)
        sigma = u * u * (3 - 2 * u)
        value = (1 - sigma) * f + sigma * F
        if not gradient:
            return value, None
        derivatives = []
        for axis in range(3):
            low = self._basis(t[..., axis], low=True)
            zero = torch.zeros_like(low[..., :1])
            db = (
                (self.count - 1)
                * (torch.cat((zero, low), -1) - torch.cat((low, zero), -1))
                / (2 * radius[..., None])
            )
            derivatives.append(
                self._contract(
                    coefficients, [db if a == axis else bases[a] for a in range(3)]
                )
            )
        df = torch.stack(derivatives, dim=-1)
        unit = offset / torch.where(rho > 0, rho, torch.ones_like(rho))[..., None]
        dsigma = 6 * u * (1 - u) / (radius - inner)
        grad = (1 - sigma[..., None]) * df + (sigma + (F - f) * dsigma)[
            ..., None
        ] * unit
        return value, grad

    def evaluate(self, poses, active):
        """Return h_RDF and dh/dq for CPU-selected active pairs, in that order."""
        torch = self.torch
        active = np.asarray(active, dtype=np.int64)
        dof = poses[self.names[0]][2].shape[1]
        self.last_active_pairs = len(active)
        if not len(active):
            return np.empty(0), np.empty((0, dof))
        packed = np.stack(
            [
                np.column_stack(
                    (poses[n][0], poses[n][1], poses[n][2][:3], poses[n][2][3:])
                )
                for n in self.names
            ]
        )
        if not np.isfinite(packed).all():
            raise ValueError("nonfinite RDF frame pose/Jacobian")
        with torch.inference_mode():
            frame = torch.as_tensor(packed, dtype=torch.float64, device=self.device)
            rows = torch.as_tensor(active, dtype=torch.long, device=self.device)
            sources, targets = self.source[rows], self.target[rows]
            source, target = frame[sources], frame[targets]
            r_source, r_target = source[..., :3], target[..., :3]
            offsets = self.points[rows] @ r_source.transpose(-1, -2)
            world = offsets + source[:, None, :, 3]
            local = (world - target[:, None, :, 3]) @ r_target
            distance, _ = self._field(local, targets)
            valid = self.positions[None] < self.lengths[rows, None]
            finite = (torch.isfinite(distance) | ~valid).all(-1)
            distance = torch.where(
                valid, distance, torch.full_like(distance, float("inf"))
            )
            k = self.options.top_k
            if k is None:
                ids = self.positions[None].expand(len(active), -1)
            else:
                # Stable sort preserves the NumPy source-index tie rule.
                ids = torch.argsort(distance, dim=-1, stable=True)[
                    :, : min(k, distance.shape[1])
                ]
            selected = distance.gather(1, ids)
            selected_local = local.gather(1, ids[..., None].expand(-1, -1, 3))
            _, gradient = self._field(selected_local, targets, gradient=True)
            world_gradient = gradient @ r_target.transpose(-1, -2)
            selected_offsets = offsets.gather(1, ids[..., None].expand(-1, -1, 3))
            target_offsets = (
                world.gather(1, ids[..., None].expand(-1, -1, 3))
                - target[:, None, :, 3]
            )
            # Jworld_point - Jtarget_origin + [world_point-target_origin]x Jomega_target.
            angular_source = source[..., 4 + dof :].transpose(-1, -2)
            angular_target = target[..., 4 + dof :].transpose(-1, -2)
            relative = source[:, None, :, 4 : 4 + dof] - target[:, None, :, 4 : 4 + dof]
            relative = relative + (
                torch.linalg.cross(
                    angular_source[:, None], selected_offsets[:, :, None], dim=-1
                )
                - torch.linalg.cross(
                    angular_target[:, None], target_offsets[:, :, None], dim=-1
                )
            ).transpose(-1, -2)
            derivative = (world_gradient[..., None] * relative).sum(-2)
            count = self.lengths[rows].clamp(max=selected.shape[1])
            beta = (
                torch.log(count.to(torch.float64).clamp(min=2))
                / self.options.softmin_error
            )
            minimum = selected.min(-1).values
            logits = -beta[:, None] * (selected - minimum[:, None])
            normalizer = torch.logsumexp(logits, dim=-1)
            weight = torch.exp(logits - normalizer[:, None])
            value = minimum - normalizer / beta - self.options.effective_margin
            grad = (weight[..., None] * derivative).sum(1)
            result = (
                torch.cat(
                    (value[:, None], grad, finite[:, None].to(torch.float64)), dim=1
                )
                .cpu()
                .numpy()
            )
        if not np.isfinite(result).all() or not np.all(result[:, -1] == 1):
            raise ValueError("nonfinite CUDA RDF distance/barrier/gradient")
        return result[:, 0].copy(), result[:, 1:-1].copy()

    def evaluate_values_batch(self, poses, active, *, pair_chunk_size=64):
        """Batch independent states/pairs without Jacobians or spatial gradients.

        Bound temporary polynomial tensors by active pair count, not by the
        number of states supplied by the caller. Keep float64/top-k/tie policy.
        """
        torch = self.torch
        if len(poses) != len(active) or pair_chunk_size < 1:
            raise ValueError("invalid RDF value batch")
        output = [np.empty(len(rows)) for rows in active]
        entries = [
            (s, j, int(row))
            for s, rows in enumerate(active)
            for j, row in enumerate(rows)
        ]
        if not entries:
            return output
        packed = np.stack(
            [[np.column_stack((p[n][0], p[n][1])) for n in self.names] for p in poses]
        )
        if not np.isfinite(packed).all():
            raise ValueError("nonfinite RDF frame pose")
        with torch.inference_mode():
            frames = torch.as_tensor(packed, dtype=torch.float64, device=self.device)
            for offset in range(0, len(entries), pair_chunk_size):
                chunk = entries[offset : offset + pair_chunk_size]
                states = torch.tensor([s for s, _, _ in chunk], device=self.device)
                rows = torch.tensor([r for _, _, r in chunk], device=self.device)
                targets = self.target[rows]
                source = frames[states, self.source[rows]]
                target = frames[states, targets]
                world = (
                    self.points[rows] @ source[..., :3].transpose(-1, -2)
                    + source[:, None, :, 3]
                )
                local = (world - target[:, None, :, 3]) @ target[..., :3]
                distance, _ = self._field(local, targets, gradient=False)
                valid = self.positions[None] < self.lengths[rows, None]
                finite = (torch.isfinite(distance) | ~valid).all(-1)
                distance = torch.where(
                    valid, distance, torch.full_like(distance, float("inf"))
                )
                k = self.options.top_k
                if k is not None:
                    ids = torch.argsort(distance, dim=-1, stable=True)[
                        :, : min(k, distance.shape[1])
                    ]
                    distance = distance.gather(1, ids)
                count = self.lengths[rows].clamp(max=distance.shape[1])
                beta = (
                    torch.log(count.to(torch.float64).clamp(min=2))
                    / self.options.softmin_error
                )
                minimum = distance.min(-1).values
                value = (
                    minimum
                    - torch.logsumexp(
                        -beta[:, None] * (distance - minimum[:, None]), dim=-1
                    )
                    / beta
                    - self.options.effective_margin
                )
                result = (
                    torch.stack((value, finite.to(torch.float64)), dim=1).cpu().numpy()
                )
                if not np.isfinite(result).all() or not np.all(result[:, 1] == 1):
                    raise ValueError("nonfinite CUDA RDF barrier")
                for (s, j, _), v in zip(chunk, result[:, 0]):
                    output[s][j] = v
        return output

"""Scope and fail-before-fit checks for the one-link deterministic refit."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh

from safe_ik.application.panda_rdf_failure_study import run_link
from safe_ik.application.panda_rdf_sign_retraining import (
    validate_protocol,
    verify_preservation,
)
from safe_ik.application.panda_rdf_training import TrainingOptions
from safe_ik.domain.panda_rdf import BernsteinRDF, ExtendedRDF


class SignRetrainingTest(unittest.TestCase):
    def protocol(self):
        return json.loads(
            (
                Path(__file__).parents[1]
                / "configs/archive/panda_rdf_link6_sign_20260930.json"
            ).read_text()
        )

    def test_approved_protocol(self):
        validate_protocol(self.protocol())

    def test_reject_expanded_scope_or_changed_fit(self):
        for key, value in [
            ("links", ["link6", "hand"]),
            ("updates", 1200),
            ("basis_count", 25),
            ("initial_state", "resume"),
            ("sign_oracle", "old"),
            ("dtype", "float32"),
            ("seed", 1),
        ]:
            p = self.protocol()
            p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(p)

    def test_reject_unpinned_reference(self):
        p = self.protocol()
        p["expected_mesh_label_array_sha256"] = {"near": "a" * 64}
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def fixture(self):
        names = [f"link{i}" for i in range(8)] + ["hand"]
        return SimpleNamespace(
            models={
                n: ExtendedRDF(
                    BernsteinRDF(np.zeros(3), 0.2, np.zeros((2,) * 3)), 0.1, 0.0
                )
                for n in names
            },
            queries={n: np.zeros((2, 3)) for n in names},
            reconstructed={n: trimesh.creation.box() for n in names},
            metadata={
                k: {n: {} for n in names}
                for k in ["validation", "maxima", "link_training"]
            },
        )

    def test_preservation_allows_only_link6_model(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b.models["link6"].rdf.weights += 1
        self.assertEqual(len(verify_preservation(a, b)), 8)
        b.models["hand"].rdf.weights += 1
        with self.assertRaisesRegex(ValueError, "preserved model"):
            verify_preservation(a, b)

    def test_preservation_rejects_changed_queries(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b.queries["link6"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "queries"):
            verify_preservation(a, b)

    def test_label_pin_failure_before_fit(self):
        module = "safe_ik.application.panda_rdf_failure_study."
        a = self.fixture()
        a.metadata["links"] = list(a.models)
        p = self.protocol()
        p["input_sha256"]["link6"] = {}
        points = {k: np.zeros((2, 3)) for k in ["near", "random", "exterior"]}
        labels = {k: np.ones(2, dtype=np.float32) * 0.02 for k in points}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch(
                    module + "read_inputs",
                    return_value=(root, points, labels, np.ones(2) * 0.02, {}),
                ),
                patch(module + "query_truth", return_value=np.ones(2) * 0.02),
                patch(module + "recursive_ridge_torch") as fit,
            ):
                with self.assertRaisesRegex(ValueError, "independent audit"):
                    run_link(
                        root, root, "link6", a, copy.deepcopy(a), p, TrainingOptions()
                    )
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()

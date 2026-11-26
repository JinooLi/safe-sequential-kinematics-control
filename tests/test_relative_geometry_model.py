"""Geometric scoring invariants and portable relative-geometry inference."""

import copy
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np
import torch

from safe_ik.domain.link_actions import LINK_ACTIONS, catalog_metadata
from safe_ik.learning.link_selection import (
    checkpoint_actions,
    load_model,
    predict_selection,
)
from safe_ik.learning.relative_geometry import (
    RelativeGeometrySuccessNet,
    _movement_features,
    capsule_clearances,
    relative_model_spec,
    validate_query_conditioning,
)
from safe_ik.learning.relative_geometry_features import (
    make_relative_geometry,
    make_relative_state,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/panda.xml"
CONFIG = ROOT / "configs/panda.yaml"


class RelativeGeometryModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = make_relative_geometry(MODEL, CONFIG)
        cls.q = np.array(
            [[0, 0, 0, -1.5, 0, 1.5, -0.7], [0.2, -0.5, 0.3, -1.8, 0.4, 1.7, -0.3]]
        )
        cls.ik = np.array(
            [
                [0.3, -0.4, 0.2, -1.8, 0.5, 1.4, -0.2],
                [-1.3, 0.8, -0.7, -0.6, -0.9, 2.5, 1.1],
            ]
        )
        cls.positions = np.array([[0.4, -0.2, 0.6], [0.3, 0.1, 0.5]])
        cls.quaternions = np.array([[1, 0, 0, 0], [0.70710678, 0, 0, 0.70710678]])
        cls.state = torch.tensor(
            make_relative_state(
                cls.q, cls.ik, cls.positions, cls.quaternions, geometry=cls.geometry
            )
        )
        cls.obstacles = torch.tensor(
            [[[0.2, 0.1, 0.6, 0.08], [-0.2, 0.3, 0.5, 0.1], [0.4, 0.2, 0.7, 0.03]]] * 2
        )

    def model(self, mode="local"):
        torch.manual_seed(53)
        return RelativeGeometrySuccessNet(
            geometry=copy.deepcopy(self.geometry), query_conditioning=mode
        ).eval()

    def test_query_experiment_matches_parameters_forward_order_and_rng(self):
        local = self.model()
        control = self.model("reordered_control")
        treatment = self.model("global_query")
        self.assertNotIn("query_conditioning", relative_model_spec(0.2, self.geometry))
        for model in (control, treatment):
            self.assertEqual(sum(p.numel() for p in model.parameters()), 47894)
            self.assertEqual(local.state_dict().keys(), model.state_dict().keys())
            for name, tensor in local.state_dict().items():
                torch.testing.assert_close(
                    tensor, model.state_dict()[name], rtol=0, atol=0
                )
        torch.testing.assert_close(
            local(self.state, self.obstacles),
            control(self.state, self.obstacles),
            rtol=0,
            atol=0,
        )
        traces, outputs, random_states = [], [], []
        for model in (control, treatment):
            trace = []
            handles = [
                module.register_forward_pre_hook(
                    lambda module, args, name=name: trace.append(
                        (name, torch.get_rng_state().clone())
                    )
                )
                for name, module in model.named_modules()
            ]
            model.train()
            torch.manual_seed(917)
            outputs.append(model(self.state, self.obstacles).detach())
            random_states.append(torch.get_rng_state())
            traces.append(trace)
            for handle in handles:
                handle.remove()
        self.assertEqual(
            [name for name, _ in traces[0]], [name for name, _ in traces[1]]
        )
        names = [name for name, _ in traces[0]]
        self.assertLess(names.index("direct_global"), names.index("link_encoder"))
        for (_, before_control), (_, before_treatment) in zip(*traces):
            self.assertTrue(torch.equal(before_control, before_treatment))
        self.assertTrue(torch.equal(*random_states))
        # Reordering changes the legacy dropout stream, but the two new modes
        # consume identical streams and leave the Direct branch untouched.
        torch.testing.assert_close(outputs[0][:, -1], outputs[1][:, -1], rtol=0, atol=0)

    def test_global_features_condition_only_link_attention_query(self):
        for mode in ("reordered_control", "global_query"):
            with self.subTest(mode=mode):
                model = self.model(mode)
                captured = {}
                handles = [
                    model.sequential_global.register_forward_hook(
                        lambda module, args, out: captured.update(
                            global_state=out.detach()
                        )
                    ),
                    model.link_encoder.register_forward_hook(
                        lambda module, args, out: captured.update(local=out.detach())
                    ),
                    model.link_attention.register_forward_pre_hook(
                        lambda module, args: captured.update(query=args[0].detach())
                    ),
                    model.link_fusion.register_forward_pre_hook(
                        lambda module, args: captured.update(fusion=args[0].detach())
                    ),
                ]
                original = model(self.state, self.obstacles)
                global_state = captured["global_state"]
                folded = (global_state[:, :32] + global_state[:, 32:]) / 2
                expected = captured["local"]
                if mode == "global_query":
                    expected = expected + folded[:, None]
                torch.testing.assert_close(
                    captured["query"].reshape(2, 5, 32), expected, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    captured["fusion"][..., :32], captured["local"], rtol=0, atol=0
                )
                old_query = captured["query"].clone()
                override = model.sequential_global.register_forward_hook(
                    lambda module, args, out: out + 1
                )
                changed = model(self.state, self.obstacles)
                torch.testing.assert_close(
                    captured["query"] - old_query,
                    torch.full_like(old_query, float(mode == "global_query")),
                    atol=2e-7,
                    rtol=1e-6,
                )
                torch.testing.assert_close(
                    original[:, -1], changed[:, -1], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    captured["fusion"][..., :32], captured["local"], rtol=0, atol=0
                )
                override.remove()
                for handle in handles:
                    handle.remove()

    def test_checkpoint_schema_rejects_ambiguous_query_conditioning(self):
        for invalid in (None, "global", [], 21):
            with self.assertRaisesRegex(ValueError, "query_conditioning"):
                validate_query_conditioning(invalid)
        for version, mode in (
            (20, "local"),
            (20, "reordered_control"),
            (20, "global_query"),
            (21, "local"),
            (21, "unknown"),
            (21, None),
        ):
            with self.subTest(version=version, mode=mode):
                specification = relative_model_spec(0.2, self.geometry)
                if mode is not None:
                    specification["query_conditioning"] = mode
                checkpoint = {
                    "schema_version": version,
                    "model": specification,
                    "output_dim": 16,
                    **catalog_metadata(),
                }
                with self.assertRaises(ValueError):
                    checkpoint_actions(checkpoint)

    def test_relative_transform_and_signed_capsule_distance(self):
        # Current axes are rotated 90 degrees about z. A world +y displacement
        # is local +x; a target with the same orientation has identity ΔR.
        current = torch.tensor([[[1.0, 2.0, 3.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]]])
        target = current.clone()
        target[..., 1] += 2
        movement = _movement_features(current, target)
        torch.testing.assert_close(
            movement[..., 9:12], torch.tensor([[[2.0, 0.0, 0.0]]])
        )
        torch.testing.assert_close(
            movement[..., 12:], torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]])
        )
        endpoints = torch.tensor(
            [[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
        )
        radii = torch.tensor([0.2, 0.1])
        obstacles = torch.tensor(
            [[[1.0, 1.0, 0.0, 0.3], [3.0, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, 0.1]]]
        )
        actual = capsule_clearances(endpoints, radii, obstacles)
        expected = torch.tensor(
            [[[0.5, 0.6, -0.3], [np.sqrt(2) - 0.4, 2.7, -0.2]]], dtype=torch.float32
        )
        torch.testing.assert_close(actual, expected)

    def test_direct_is_exactly_independent_of_selected_ik(self):
        for mode in ("local", "reordered_control", "global_query"):
            with self.subTest(mode=mode):
                model = self.model(mode)
                original = model(self.state, self.obstacles)
                changed = torch.tensor(
                    make_relative_state(
                        self.q,
                        self.ik[::-1].copy(),
                        self.positions,
                        self.quaternions,
                        geometry=self.geometry,
                    )
                )
                replacement = model(changed, self.obstacles)
                torch.testing.assert_close(
                    original[:, -1], replacement[:, -1], rtol=0, atol=0
                )
                self.assertGreater(
                    float(
                        (original[:, :-1] - replacement[:, :-1]).detach().abs().max()
                    ),
                    1e-6,
                )
                state = self.state.clone().requires_grad_()
                model(state, self.obstacles)[:, -1].sum().backward()
                target_start = 122 + 6 * model.capsule_count
                for section in (
                    slice(7, 14),
                    slice(68, 113),
                    slice(target_start, None),
                ):
                    self.assertEqual(float(state.grad[:, section].abs().sum()), 0.0)

    def test_obstacle_order_padding_and_empty_mask_are_invariant(self):
        for mode in ("local", "reordered_control", "global_query"):
            with self.subTest(mode=mode):
                model = self.model(mode)
                expected = model(self.state, self.obstacles)
                torch.testing.assert_close(
                    model(self.state, self.obstacles[:, [2, 0, 1]]),
                    expected,
                    atol=2e-7,
                    rtol=1e-6,
                )
                padded = torch.cat(
                    (self.obstacles, torch.full((2, 2, 4), float("nan"))), dim=1
                )
                mask = torch.tensor([[1, 1, 1, 0, 0]] * 2, dtype=torch.bool)
                torch.testing.assert_close(
                    model(self.state, padded, mask), expected, atol=2e-7, rtol=1e-6
                )
                no_obstacles = model(self.state, self.obstacles[:, :0])
                all_masked = model(self.state, padded, torch.zeros_like(mask))
                self.assertTrue(torch.isfinite(all_masked).all())
                torch.testing.assert_close(
                    all_masked, no_obstacles, atol=2e-7, rtol=1e-6
                )
                state = self.state.clone().requires_grad_()
                padded.requires_grad_()
                model(state, padded, mask).square().mean().backward()
                self.assertTrue(torch.isfinite(state.grad).all())
                self.assertTrue(torch.isfinite(padded.grad).all())
                self.assertEqual(float(padded.grad[:, 3:].abs().sum()), 0)
                self.assertTrue(
                    all(
                        torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )

    def test_ordered_shared_action_head_and_all_body_context(self):
        model = self.model()
        captured = {}
        handles = [
            model.link_fusion.register_forward_hook(
                lambda m, args, out: captured.update(tokens=out.detach())
            ),
            model.sequential_head.register_forward_pre_hook(
                lambda m, args: captured.update(head=args[0].detach())
            ),
            model.direct_global.register_forward_pre_hook(
                lambda m, args: captured.update(direct=args[0].detach())
            ),
        ]
        result = model(self.state, self.obstacles)
        for handle in handles:
            handle.remove()
        self.assertEqual(result.shape, (2, 16))
        self.assertEqual(model.actions, LINK_ACTIONS)
        self.assertEqual(model.link_attention.num_heads, 1)
        self.assertEqual(model.ee_attention.num_heads, 1)
        self.assertEqual(model.sequential_head[-1].out_features, 1)
        for index, action in enumerate(LINK_ACTIONS[:-1]):
            torch.testing.assert_close(
                captured["head"][:, index, 64:96], captured["tokens"][:, action[0] - 2]
            )
            if len(action) == 2:
                torch.testing.assert_close(
                    captured["head"][:, index, 96:128],
                    captured["tokens"][:, action[1] - 2],
                )
            else:
                self.assertEqual(
                    float(captured["head"][:, index, 96:128].abs().sum()), 0
                )
            torch.testing.assert_close(
                captured["head"][:, index, -2:],
                torch.tensor([[1.0, float(len(action) == 2)]] * 2),
            )
        self.assertEqual(captured["direct"].shape, (2, 17 + model.capsule_count))
        split = 122 + 6 * model.capsule_count
        distances = capsule_clearances(
            self.state[:, 122:split].reshape(2, model.capsule_count, 2, 3),
            model.capsule_radii,
            self.obstacles,
        )
        torch.testing.assert_close(captured["direct"][:, 16:-1], distances.amin(-1))

    def test_portable_checkpoint_and_online_features_match_training(self):
        for mode in ("local", "reordered_control", "global_query"):
            with self.subTest(mode=mode):
                self.check_portable_checkpoint(mode)

    def check_portable_checkpoint(self, mode):
        model = self.model(mode)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "before"
            directory.mkdir()
            shutil.copyfile(MODEL, directory / "panda.xml")
            shutil.copyfile(CONFIG, directory / "controller.yaml")
            geometry = make_relative_geometry(
                directory / "panda.xml",
                directory / "controller.yaml",
                checkpoint_directory=directory,
            )
            checkpoint = {
                "schema_version": 20 if mode == "local" else 21,
                "model": relative_model_spec(0.2, geometry, query_conditioning=mode),
                "relative_geometry": geometry,
                "output_dim": 16,
                **catalog_metadata(),
                "optimizer": "adamw",
                "weight_decay": 0.0001,
                "selection_loss_weight": 1.0,
                "bce_loss_weight": 1.0,
                "negative_bce_weight": 1.0,
                "selection_metric": "validation_selected_success_rate",
                "state_dict": model.state_dict(),
            }
            torch.save(checkpoint, directory / "model.pt")
            moved = Path(temporary) / "after"
            shutil.move(directory, moved)
            loaded = load_model(moved / "model.pt")
            self.assertEqual(loaded.query_conditioning, mode)
            expected = model(self.state, self.obstacles)
            torch.testing.assert_close(
                loaded(self.state, self.obstacles), expected, rtol=0, atol=0
            )
            result = predict_selection(
                loaded,
                self.q[0],
                self.ik[0],
                self.positions[0],
                self.quaternions[0],
                self.obstacles[0].numpy(),
            )
            np.testing.assert_allclose(
                result["probabilities"],
                expected[0].sigmoid().detach().numpy(),
                atol=1e-7,
                rtol=1e-6,
            )
            self.assertEqual(result["action_index"], int(expected[0].argmax()))
            # Inference requires embedded capsule definitions and the geometry,
            # but not a runtime controller configuration or RDF model.
            (moved / "controller.yaml").unlink()
            load_model(moved / "model.pt")
            (moved / "panda.xml").write_text(
                (moved / "panda.xml").read_text() + "\n<!-- changed -->\n"
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_model(moved / "model.pt")
            load_model(moved / "model.pt", fk_model_path=MODEL)
        bypass = Mock(side_effect=AssertionError("missing IK must bypass network"))
        self.assertEqual(
            predict_selection(bypass, None, None, None, None, None)["reason"],
            "no_validated_ik",
        )
        bypass.assert_not_called()


if __name__ == "__main__":
    unittest.main()

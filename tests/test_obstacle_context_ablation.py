"""Matched Original23 obstacle-context ablation and checkpoint semantics."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from safe_ik.domain.link_actions import catalog_metadata
from safe_ik.learning.link_selection import (
    MODEL_SPEC,
    LinkSuccessNet,
    checkpoint_actions,
    load_model,
    model_spec,
    validate_obstacle_context,
)


class ObstacleContextAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.state = torch.randn(4, 23)
        self.obstacles = torch.randn(4, 3, 4)

    def model(self, context):
        torch.manual_seed(11)
        return LinkSuccessNet(mlp_dropout=0.2, obstacle_context=context)

    def test_initial_tensors_and_module_dropout_order_are_matched(self):
        full, zero = self.model("full"), self.model("zero")
        self.assertEqual(sum(p.numel() for p in full.parameters()), 24624)
        self.assertEqual(sum(p.numel() for p in zero.parameters()), 24624)
        self.assertEqual(full.state_dict().keys(), zero.state_dict().keys())
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, zero.state_dict()[key], rtol=0, atol=0)
        traces, end_states = [], []
        for model in (full, zero):
            trace = []
            handles = [
                module.register_forward_pre_hook(
                    lambda module, args, name=name: trace.append(
                        (name, torch.get_rng_state().clone())
                    )
                )
                for name, module in model.named_modules()
            ]
            torch.manual_seed(20261124)
            model(self.state, self.obstacles)
            traces.append(trace)
            end_states.append(torch.get_rng_state())
            for handle in handles:
                handle.remove()
        self.assertEqual([n for n, _ in traces[0]], [n for n, _ in traces[1]])
        for (_, left), (_, right) in zip(*traces):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(*end_states))

    def test_zero_context_removes_obstacle_dependence_but_keeps_robot_gradient(self):
        zero = self.model("zero").eval()
        state = self.state.clone().requires_grad_()
        obstacles = self.obstacles.clone().requires_grad_()
        logits = zero(state, obstacles)
        torch.testing.assert_close(logits, zero(state, obstacles + 10), rtol=0, atol=0)
        padded = torch.cat((obstacles, torch.full((4, 2, 4), float("nan"))), dim=1)
        mask = torch.tensor([[1, 1, 1, 0, 0]] * 4, dtype=torch.bool)
        for other, other_mask in (
            (padded, mask),
            (padded, torch.zeros_like(mask)),
            (obstacles[:, :0], None),
        ):
            torch.testing.assert_close(
                logits, zero(state, other, other_mask), rtol=0, atol=0
            )
        self.assertFalse(torch.equal(logits, zero(state + 1, obstacles)))
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertGreater(float(state.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(obstacles.grad, torch.zeros_like(obstacles.grad)))
        for name, parameter in zero.named_parameters():
            if name.startswith(
                (
                    "obstacle_encoder.",
                    "query_projection.",
                    "cross_attention.",
                    "context_projection.",
                )
            ):
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(float(parameter.grad.abs().sum()), 0, name)
        full = self.model("full").eval()
        self.assertFalse(
            torch.equal(full(state, obstacles), full(state, obstacles + 10))
        )

    def test_default_specification_and_restricted_ablation_modes(self):
        self.assertEqual(model_spec(), MODEL_SPEC)
        self.assertEqual(model_spec(0.2), {**MODEL_SPEC, "mlp_dropout": 0.2})
        self.assertEqual(
            model_spec(0.2, obstacle_context="zero"),
            {**MODEL_SPEC, "mlp_dropout": 0.2, "obstacle_context": "zero"},
        )
        for invalid in (None, False, [], "none", "robot_only"):
            with self.assertRaisesRegex(ValueError, "obstacle_context"):
                validate_obstacle_context(invalid)
        for constructor in (LinkSuccessNet, model_spec):
            with self.assertRaisesRegex(ValueError, "23D"):
                constructor(ik_target_features=True, obstacle_context="zero")

    def test_checkpoint_roundtrip_and_version_mode_conflicts(self):
        checkpoint = {
            "output_dim": 16,
            **catalog_metadata(),
            "optimizer": "adamw",
            "weight_decay": 0.0001,
            "bce_loss_weight": 1.0,
            "selection_loss_weight": 1.0,
            "negative_bce_weight": 1.0,
            "selection_metric": "validation_selected_success_rate",
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            for mode, version in (("full", 17), ("zero", 22)):
                model = self.model(mode).eval()
                saved = {
                    **checkpoint,
                    "schema_version": version,
                    "model": model_spec(0.2, obstacle_context=mode),
                    "state_dict": model.state_dict(),
                }
                torch.save(saved, path)
                loaded = load_model(path)
                self.assertEqual(loaded.obstacle_context, mode)
                torch.testing.assert_close(
                    model(self.state, self.obstacles),
                    loaded(self.state, self.obstacles),
                    rtol=0,
                    atol=0,
                )
            for mode in (None, "full", "other", False):
                invalid = copy.deepcopy(saved)
                if mode is None:
                    invalid["model"].pop("obstacle_context")
                else:
                    invalid["model"]["obstacle_context"] = mode
                torch.save(invalid, path)
                with self.assertRaisesRegex(ValueError, "schema 22"):
                    load_model(path)
            for version in (11, 12, 13, 16, 17, 18, 19, 20, 21):
                with self.subTest(version=version):
                    invalid = {**saved, "schema_version": version}
                    with self.assertRaisesRegex(ValueError, "schema 22"):
                        checkpoint_actions(invalid)
            invalid = copy.deepcopy(saved)
            invalid["model"]["ik_target_features"] = {"enabled": True}
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, "selected model"):
                load_model(path)


if __name__ == "__main__":
    unittest.main()

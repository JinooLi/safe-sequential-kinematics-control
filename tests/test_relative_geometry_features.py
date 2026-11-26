"""FK features retain physical geometry and exact asset provenance."""

import copy
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
import yaml

from safe_ik.learning.relative_geometry_features import (
    RELATIVE_LINK_FRAMES,
    compute_relative_geometry_features,
    make_relative_geometry,
    make_relative_state,
    relative_geometry_state_dim,
    validate_relative_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/panda.xml"
CONFIG = ROOT / "configs/panda.yaml"


class RelativeGeometryFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = np.array(
            [
                [0.1, -0.3, 0.2, -1.7, 0.4, 1.2, -0.7],
                [-1.1, 0.6, -0.8, -0.9, -0.5, 2.1, 1.0],
            ]
        )
        cls.target = np.array(
            [
                [0.5, 0.2, -0.3, -2.0, 0.1, 1.8, -0.1],
                [-0.7, -0.2, 0.5, -1.4, 0.6, 1.0, 0.4],
            ]
        )
        cls.geometry = make_relative_geometry(MODEL, CONFIG)

    def test_current_target_poses_and_every_capsule_match_independent_mujoco(self):
        geometry = self.geometry
        features = compute_relative_geometry_features(
            self.current, self.target, MODEL, geometry
        )
        self.assertEqual(features.shape, (2, 219))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(relative_geometry_state_dim(geometry), 242)
        self.assertEqual(geometry["link_capsule_indices"], [2, 3, 4, 5, 6])
        self.assertEqual(geometry["physical_distance_safety_margin"], 0)
        source_capsules = yaml.safe_load(CONFIG.read_text())["safety"]["capsules"]
        self.assertEqual(geometry["capsules"], source_capsules)
        # Capsule radii are not inflated by the controller's 5 cm margin.
        self.assertEqual(geometry["capsule_radii"][2], 0.025)
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        data = mujoco.MjData(model)

        def pose(frame):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame)
            if body >= 0:
                return data.xpos[body], data.xmat[body].reshape(3, 3)
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame)
            self.assertGreaterEqual(site, 0)
            return data.site_xpos[site], data.site_xmat[site].reshape(3, 3)

        for row in range(2):
            for target, configurations in ((False, self.current), (True, self.target)):
                data.qpos[:] = configurations[row]
                mujoco.mj_forward(model, data)
                base = 45 if target else 0
                for index, frame in enumerate(RELATIVE_LINK_FRAMES):
                    position, rotation = pose(frame)
                    np.testing.assert_allclose(
                        features[row, base + index * 9 : base + (index + 1) * 9],
                        np.r_[position, rotation[:, 0], rotation[:, 1]],
                        atol=6e-8,
                        rtol=0,
                    )
                if not target:
                    position, rotation = pose(geometry["ee_frame"])
                    np.testing.assert_allclose(
                        features[row, 90:99],
                        np.r_[position, rotation[:, 0], rotation[:, 1]],
                        atol=6e-8,
                        rtol=0,
                    )
                base = 159 if target else 99
                for index, capsule in enumerate(source_capsules):
                    for end_index, end in enumerate(("start", "end")):
                        position, rotation = pose(capsule[f"{end}_frame"])
                        expected = position + rotation @ capsule[f"{end}_offset"]
                        offset = base + 6 * index + 3 * end_index
                        np.testing.assert_allclose(
                            features[row, offset : offset + 3],
                            expected,
                            atol=6e-8,
                            rtol=0,
                        )
        packed = make_relative_state(
            self.current,
            self.target,
            np.zeros((2, 3)),
            np.tile([1, 0, 0, 0], (2, 1)),
            geometry=geometry,
        )
        self.assertEqual(packed.shape, (2, 242))
        np.testing.assert_array_equal(packed[:, 23:], features)
        np.testing.assert_array_equal(packed[:, :7], self.current.astype(np.float32))
        np.testing.assert_array_equal(packed[:, 7:14], self.target.astype(np.float32))

    def test_portable_snapshot_uses_embedded_capsules_and_rejects_modified_model(self):
        with TemporaryDirectory() as temporary:
            original = Path(temporary) / "original"
            original.mkdir()
            shutil.copyfile(MODEL, original / "panda.xml")
            shutil.copyfile(CONFIG, original / "controller.yaml")
            geometry = make_relative_geometry(
                original / "panda.xml",
                original / "controller.yaml",
                checkpoint_directory=original,
            )
            self.assertEqual(geometry["model_path"], "panda.xml")
            moved = Path(temporary) / "moved"
            shutil.move(original, moved)
            # Deployment needs no original controller YAML or RDF artifact.
            (moved / "controller.yaml").unlink()
            resolved = validate_relative_geometry(geometry, checkpoint_directory=moved)
            self.assertEqual(resolved, moved / "panda.xml")
            expected = compute_relative_geometry_features(
                self.current, self.target, MODEL, self.geometry
            )
            np.testing.assert_array_equal(
                compute_relative_geometry_features(
                    self.current, self.target, resolved, geometry
                ),
                expected,
            )
            changed = moved / "panda.xml"
            changed.write_text(changed.read_text() + "\n<!-- changed -->\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_relative_geometry(geometry, checkpoint_directory=moved)
            self.assertEqual(
                validate_relative_geometry(geometry, model_path=MODEL), MODEL
            )
            with self.assertRaisesRegex(ValueError, "checkpoint directory"):
                validate_relative_geometry(geometry)

    def test_capsule_identity_and_metadata_integrity_are_not_silently_inferred(self):
        for key, value in (
            ("link_capsule_indices", [6, 5, 4, 3, 2]),
            ("link_frames", list(reversed(RELATIVE_LINK_FRAMES))),
            ("physical_distance_safety_margin", 0.05),
            ("state_dim", 241),
        ):
            changed = copy.deepcopy(self.geometry)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_relative_geometry(changed)
        changed = copy.deepcopy(self.geometry)
        changed["capsules"][0]["start_offset"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_relative_geometry(changed)
        source = yaml.safe_load(CONFIG.read_text())
        with TemporaryDirectory() as temporary:
            config = Path(temporary) / "controller.yaml"
            for duplicate in (False, True):
                altered = copy.deepcopy(source)
                if duplicate:
                    altered["safety"]["capsules"].append(
                        altered["safety"]["capsules"][2]
                    )
                else:
                    altered["safety"]["capsules"][2]["name"] = "unidentified"
                config.write_text(yaml.safe_dump(altered))
                with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                    make_relative_geometry(MODEL, config)

    def test_fk_rejects_misaligned_or_nonfinite_scene_inputs(self):
        for current, target in (
            (np.zeros(7), np.zeros(7)),
            (np.zeros((2, 7)), np.zeros((1, 7))),
            (np.full((1, 7), np.nan), np.zeros((1, 7))),
            (np.zeros((1, 7)), np.full((1, 7), np.inf)),
        ):
            with self.subTest(shape=current.shape), self.assertRaises(ValueError):
                compute_relative_geometry_features(
                    current, target, MODEL, self.geometry
                )


if __name__ == "__main__":
    unittest.main()

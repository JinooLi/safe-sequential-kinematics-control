"""Config relocation keeps CLI defaults usable without starting any training."""

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from safe_ik.application import refit_panda_rdf, train_panda_rdf
from safe_ik.application.panda_rdf_training import TrainingOptions
from safe_ik.application.rdf_exterior_sampling import ExteriorSamplingOptions

ROOT = Path(__file__).resolve().parents[1]


class ArgumentsInspected(Exception):
    """Stop immediately after parsing, before reading inputs or writing outputs."""


class ConfigLayoutTest(unittest.TestCase):
    def test_training_clis_keep_bp10_default_at_new_location(self):
        parse = argparse.ArgumentParser.parse_args
        expected = ROOT / "configs/training/panda_rdf_training.yaml"
        for cli, arguments in (
            (train_panda_rdf, []),
            (refit_panda_rdf, ["--inputs", "unused", "--output", "unused"]),
        ):

            def inspect(parser):
                args = parse(parser, arguments)
                path = Path(args.training_config)
                self.assertEqual(path, expected)
                options = TrainingOptions(**yaml.safe_load(path.read_text()))
                self.assertEqual((options.basis_count, options.updates), (10, 200))
                raise ArgumentsInspected

            with (
                self.subTest(cli=cli.__name__),
                patch.object(argparse.ArgumentParser, "parse_args", inspect),
                self.assertRaises(ArgumentsInspected),
            ):
                cli.main()

    def test_full_training_and_exterior_configs_remain_compatible(self):
        directory = ROOT / "configs/training"
        training = TrainingOptions(
            **yaml.safe_load(
                (directory / "panda_rdf_training_full_bp24.yaml").read_text()
            )
        )
        exterior = ExteriorSamplingOptions(
            **yaml.safe_load(
                (directory / "panda_rdf_exterior_sampling.yaml").read_text()
            )
        )
        self.assertEqual((training.basis_count, training.updates), (24, 800))
        self.assertEqual(training.pool_size, exterior.pool_size)
        self.assertEqual(
            (exterior.near_batch, exterior.random_batch, exterior.exterior_batch),
            (512, 128, 640),
        )
        self.assertEqual(
            training.near_batch + training.random_batch,
            exterior.near_batch + exterior.random_batch + exterior.exterior_batch,
        )


if __name__ == "__main__":
    unittest.main()

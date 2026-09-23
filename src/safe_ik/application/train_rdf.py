"""Train and validate a 2D Bernstein RDF for the configured planar arm."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..domain.rdf import PlanarBernsteinRobot, capsule_sdf_local
from .paths import default_resource
from .planar_configuration import load_controller_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_resource("configs/planar.yaml"))
    parser.add_argument("--output", default="models/planar_bernstein_rdf.npz")
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_controller_config(args.config)
    safety = config.safety
    robot = PlanarBernsteinRobot.train_from_capsules(
        safety.link_lengths,
        safety.link_radius,
        degree=args.degree,
        sample_count=args.samples,
        rng=np.random.default_rng(args.seed),
    )
    robot.save(args.output)

    # Report the validation error used to select the conservative safety margin.
    rng = np.random.default_rng(args.seed + 1)
    errors = []
    for length, model in zip(robot.link_lengths, robot.models):
        points = rng.uniform(model.bounds[:, 0], model.bounds[:, 1], size=(20000, 2))
        truth = capsule_sdf_local(points, length, robot.link_radius)
        prediction, _ = model.query(points)
        errors.append(np.abs(prediction - truth))
    error = np.concatenate(errors)
    print(f"Saved RDF model: {Path(args.output)}")
    print(f"Validation MAE: {error.mean():.6f} m")
    print(f"Validation max error: {error.max():.6f} m")
    print("Add at least the max error to safety.safe_distance before enabling RDF.")


if __name__ == "__main__":
    main()

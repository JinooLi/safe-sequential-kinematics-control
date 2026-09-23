"""Render a 2D robot distance field, its surface, and safety level set."""

from __future__ import annotations

import argparse

import numpy as np

from ..domain.rdf import PlanarBernsteinRobot
from ..domain.safety import PlanarCapsuleRobot, obstacles_from_config
from ..visualization.rdf import render_distance_field
from .paths import default_resource
from .planar_configuration import load_controller_config


def build_distance_model(config, representation: str):
    """Build the requested SDF implementation from controller configuration."""
    safety = config.safety
    if representation == "configured":
        representation = safety.representation
    if representation == "capsule":
        return PlanarCapsuleRobot(safety.link_lengths, safety.link_radius)
    if representation == "bernstein":
        return PlanarBernsteinRobot.load(safety.rdf_model_path)
    raise ValueError("representation must be configured, capsule, or bernstein")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_resource("configs/planar.yaml"))
    parser.add_argument(
        "--representation",
        default="configured",
        choices=("configured", "capsule", "bernstein"),
    )
    parser.add_argument(
        "--q", type=float, nargs="+", default=None, help="Joint angles [rad]"
    )
    parser.add_argument("--resolution", type=int, default=220)
    parser.add_argument("--output", default="outputs/distance_field.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    config = load_controller_config(args.config)
    q = np.zeros(config.dof) if args.q is None else np.asarray(args.q, dtype=float)
    if q.shape != (config.dof,):
        raise ValueError(f"--q must provide exactly {config.dof} joint angles")
    robot = build_distance_model(config, args.representation)
    obstacles = obstacles_from_config(
        [
            {"pos": item.position, "radius": item.radius}
            for item in config.obstacles
            if item.enabled
        ]
    )
    render_distance_field(
        robot,
        obstacles,
        config.safety.safe_distance,
        q,
        resolution=args.resolution,
        output_path=args.output,
        show=args.show,
    )
    print(f"Saved distance-field visualization: {args.output}")


if __name__ == "__main__":
    main()

"""CLI for planar direct/sequential experiments and saved-result plots."""

import argparse
from pathlib import Path

from .paths import project_root


def main():
    from ..experiments.runner import run_comparison
    from ..visualization.comparison import visualize_all_cases, visualize_case

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run randomized direct/sequential ablation")
    run.add_argument("--root", default=str(project_root()))
    run.add_argument("--config", default="configs/planar.yaml")
    run.add_argument("--mujoco", default="assets/robots/planar.xml")
    run.add_argument("--urdf", default="assets/robots/planar.urdf")
    run.add_argument("--run-name")
    run.add_argument("--cases", type=int, help="override sequential_experiment.cases")
    run.add_argument(
        "--workers", type=int, help="override sequential_experiment.workers"
    )
    run.add_argument(
        "--duration", type=float, help="override sequential_experiment.duration [s]"
    )
    run.set_defaults(func=run_comparison)
    visualize = subparsers.add_parser(
        "visualize", help="render one saved comparison case"
    )
    visualize.add_argument("--results-dir", type=Path, required=True)
    visualize.add_argument("--case-id", type=int, required=True)
    visualize.add_argument("--output", type=Path)
    visualize.set_defaults(
        func=lambda args: visualize_case(args.results_dir, args.case_id, args.output)
    )
    visualize_all = subparsers.add_parser(
        "visualize-all", help="render every saved comparison case"
    )
    visualize_all.add_argument("--results-dir", type=Path, required=True)
    visualize_all.add_argument("--output-dir", type=Path)
    visualize_all.set_defaults(
        func=lambda args: visualize_all_cases(args.results_dir, args.output_dir)
    )
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

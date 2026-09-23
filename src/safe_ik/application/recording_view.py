"""Render or export saved Panda traces and planar Recorder data without rerunning."""

import argparse
from pathlib import Path

import numpy as np

from ..logging.recorder import Recorder
from ..visualization.panda import save_panda_control_plot
from .panda_results import PandaControlTrace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="robot", required=True)
    for robot in ("panda", "planar"):
        command = subparsers.add_parser(robot)
        command.add_argument("--trace", type=Path, required=True)
        command.add_argument("--output", type=Path)
        command.add_argument("--show", action="store_true")
        if robot == "planar":
            command.add_argument(
                "--action", choices=("plot", "gui", "csv"), default="plot"
            )
    args = parser.parse_args()
    if args.robot == "panda":
        with np.load(args.trace, allow_pickle=False) as data:
            trace = PandaControlTrace(
                time=data["time"],
                clf_value=data["clf_value"],
                clf_slack=data["clf_slack"],
                stage_index=data["stage_index"],
                stage_names=tuple(data["stage_names"].tolist()),
                barrier_labels=tuple(data["barrier_labels"].tolist()),
                barrier_values=data["barrier_values"],
            )
        output = args.output or args.trace.with_name(args.trace.stem + "_control.png")
        if not save_panda_control_plot(trace, output, show=args.show):
            parser.error("trace contains no control samples")
    else:
        recorder = Recorder.from_npz(args.trace)
        if args.action == "gui":
            recorder.show_gui()
            return
        if args.action == "csv":
            output = args.output or args.trace.with_suffix(".csv")
            recorder.save_csv(output)
        else:
            output = args.output or args.trace.with_name(
                args.trace.stem + "_control.png"
            )
            recorder.plot(output, show=args.show)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

"""Replay completed Panda rollouts, including their last integrated state."""

import numpy as np

from ..domain.panda_types import PandaRunResult


def replay_panda_result(plant, result: PandaRunResult) -> None:
    # Trace q/time are sampled BEFORE each integration. Append final_q/time so
    # a successful last step is displayed too. Failures keep the stopped pose.
    trace = result.trace
    if trace is not None and trace.q is not None and trace.time.size:
        timestamps = np.r_[trace.time, plant.time]
        positions = np.vstack((trace.q, result.final_q))
    else:
        timestamps = np.array([plant.time])
        positions = result.final_q[None, :]
    plant.replay(timestamps, positions)

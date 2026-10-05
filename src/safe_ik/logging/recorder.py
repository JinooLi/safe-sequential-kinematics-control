import csv
import os

import numpy as np

from ..domain.types import SimulationStep
from .labels import component_label


class Recorder:
    """시뮬레이션 데이터를 메모리에 기록하고 NPZ와 CSV로 내보낸다.

    그래프와 GUI는 실행 계층이 visualization.recording에 arrays()와
    barrier_labels를 전달하여 표시한다.

    Usage:
        recorder = Recorder()
        recorder.record(step)
        recorder.save_npz("outputs/sim_data.npz")
        recorder.save_csv("outputs/sim_data.csv")
    """

    def __init__(self):
        self.barrier_labels: tuple[str, ...] = ()
        self._data = {
            "t": [],
            "q": [],
            "dq": [],
            "ee": [],
            "ee_yaw": [],
            "u": [],
            "clf": [],
            "barrier": [],
            "active_link": [],
            "clf_slack": [],
            "qp_solve_time": [],
        }

    def record(self, step: SimulationStep):
        """Record one typed simulation step and its named safety diagnostics."""
        labels = step.safety.barrier_labels
        if self.barrier_labels and self.barrier_labels != labels:
            raise ValueError("barrier label order changed during recording")
        self.barrier_labels = labels
        self._data["t"].append(float(step.time))
        self._data["q"].append(np.asarray(step.q, dtype=float).copy())
        self._data["dq"].append(np.asarray(step.dq, dtype=float).copy())
        self._data["ee"].append(np.asarray(step.end_effector, dtype=float).copy())
        self._data["ee_yaw"].append(float(step.end_effector_yaw))
        self._data["u"].append(np.asarray(step.command, dtype=float).copy())
        self._data["clf"].append(step.safety.clf_value)
        self._data["barrier"].append(step.safety.barrier_values.copy())
        self._data["active_link"].append(step.active_link)
        self._data["clf_slack"].append(step.clf_slack)
        self._data["qp_solve_time"].append(step.qp_solve_time)

    def arrays(self):
        """기록된 데이터를 numpy array dictionary로 반환한다."""
        out = {}
        for key, values in self._data.items():
            if not values or values[0] is None:
                continue
            out[key] = np.asarray(values)
        return out

    @classmethod
    def from_npz(cls, path):
        """save_npz()로 저장한 데이터를 다시 Recorder로 불러온다.

        Args:
            path: .npz 데이터 파일 경로.

        Returns:
            Recorder: GUI 표시나 재저장에 사용할 수 있는 recorder 객체.

        Usage:
            recorder = Recorder.from_npz("outputs/sim_data.npz")
            arrays = recorder.arrays()
        """
        recorder = cls()
        loaded = np.load(path)
        if "barrier_labels" in loaded:
            recorder.barrier_labels = tuple(
                str(label) for label in loaded["barrier_labels"]
            )
        for key in recorder._data:
            recorder._data[key] = []
            if key not in loaded:
                continue
            for value in loaded[key]:
                if key == "t":
                    recorder._data[key].append(float(value))
                else:
                    recorder._data[key].append(np.asarray(value, dtype=float))
        return recorder

    def save_npz(self, path):
        """기록된 데이터를 .npz 파일로 저장한다."""
        self._ensure_parent_dir(path)
        np.savez(
            path,
            **self.arrays(),
            barrier_labels=np.asarray(self.barrier_labels, dtype=str),
        )

    def save_csv(self, path):
        """기록된 데이터를 사람이 읽기 쉬운 CSV 파일로 저장한다.

        Vector 값은 q_0, q_1처럼 column을 나누어 저장한다.
        """
        arrays = self.arrays()
        if "t" not in arrays:
            raise ValueError("No recorded data to save.")

        self._ensure_parent_dir(path)
        headers = ["t"]
        vector_keys = [
            key
            for key in (
                "q",
                "dq",
                "ee",
                "ee_yaw",
                "u",
                "clf",
                "barrier",
                "active_link",
                "clf_slack",
                "qp_solve_time",
            )
            if key in arrays
        ]
        for key in vector_keys:
            width = self._vector_width(arrays[key])
            headers.extend(self._component_label(key, i) for i in range(width))

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i, t in enumerate(arrays["t"]):
                row = [t]
                for key in vector_keys:
                    row.extend(np.ravel(arrays[key][i]))
                writer.writerow(row)

    @staticmethod
    def _ensure_parent_dir(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _vector_width(values):
        values = np.asarray(values)
        if values.ndim == 1:
            return 1
        return values.shape[1]

    def _component_label(self, key, index):
        return component_label(key, index, self.barrier_labels)

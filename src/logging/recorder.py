import csv
import os

import numpy as np

from ..domain.types import SimulationStep


class Recorder:
    """시뮬레이션 데이터를 메모리에 기록하고 파일/그래프로 내보낸다.

    Usage:
        recorder = Recorder()
        recorder.record(step)
        recorder.save_npz("outputs/sim_data.npz")
        recorder.save_csv("outputs/sim_data.csv")
        recorder.plot("outputs/sim_plot.png")
        recorder.show_gui()
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
            recorder.show_gui()
        """
        recorder = cls()
        loaded = np.load(path)
        if "barrier_labels" in loaded:
            recorder.barrier_labels = tuple(str(label) for label in loaded["barrier_labels"])
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
            for key in ("q", "dq", "ee", "ee_yaw", "u", "clf", "barrier")
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

    def plot(self, path=None, show=False):
        """기록된 데이터를 matplotlib으로 시각화한다.

        Args:
            path: 지정하면 figure를 이미지 파일로 저장한다. 예: "outputs/sim_plot.png".
            show: True이면 matplotlib 창을 띄운다.

        Returns:
            matplotlib Figure 객체.
        """
        self._configure_matplotlib_cache()

        arrays = self.arrays()
        if "t" not in arrays:
            raise ValueError("No recorded data to plot.")

        plot_keys = [
            key
            for key in ("q", "dq", "ee", "ee_yaw", "u", "clf", "barrier")
            if key in arrays
        ]
        if show:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                len(plot_keys), 1, sharex=True, figsize=(10, 2.4 * len(plot_keys))
            )
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(10, 2.4 * len(plot_keys)))
            FigureCanvasAgg(fig)
            axes = fig.subplots(len(plot_keys), 1, sharex=True)

        if len(plot_keys) == 1:
            axes = [axes]

        t = arrays["t"]
        for ax, key in zip(axes, plot_keys):
            values = np.asarray(arrays[key])
            if values.ndim == 1:
                ax.plot(t, values, label=key)
            else:
                for i in range(values.shape[1]):
                    ax.plot(t, values[:, i], label=self._component_label(key, i))
            if key == "barrier":
                ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="boundary")
            ax.set_ylabel(key)
            ax.grid(True)
            ax.legend(loc="best")

        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()

        if path is not None:
            self._ensure_parent_dir(path)
            fig.savefig(path, dpi=160)
        if show:
            plt.show()

        return fig

    def show_gui(self):
        """기록된 데이터를 matplotlib GUI로 탐색한다.

        왼쪽 버튼으로 표시할 데이터 그룹(q, dq, ee, u 등)을 바꿀 수 있고,
        아래 slider로 시간 위치를 움직이면 현재 시각의 값을 오른쪽 패널에서
        확인할 수 있다.

        Usage:
            recorder.show_gui()

            # 저장된 데이터를 나중에 보고 싶을 때
            Recorder.from_npz("outputs/sim_data.npz").show_gui()
        """
        self._configure_matplotlib_cache()
        self._select_interactive_backend()
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        arrays = self.arrays()
        if "t" not in arrays:
            raise ValueError("No recorded data to show.")

        plot_keys = [
            key
            for key in ("q", "dq", "ee", "ee_yaw", "u", "clf", "barrier")
            if key in arrays
        ]
        if not plot_keys:
            raise ValueError("No vector data to show.")

        t = arrays["t"]
        state = {"key": plot_keys[0], "index": 0}

        fig = plt.figure(figsize=(12, 7))
        ax_plot = fig.add_axes([0.25, 0.24, 0.70, 0.68])
        ax_slider = fig.add_axes([0.25, 0.10, 0.70, 0.04])
        ax_value = fig.add_axes([0.03, 0.10, 0.17, 0.20])
        ax_value.axis("off")

        slider = Slider(
            ax=ax_slider,
            label="time [s]",
            valmin=float(t[0]),
            valmax=float(t[-1]),
            valinit=float(t[0]),
        )

        lines = []

        def nearest_index(time_value):
            return int(np.argmin(np.abs(t - time_value)))

        def update_value_panel(key, sample_index):
            values = np.asarray(arrays[key][sample_index])
            text = [f"{key} @ {t[sample_index]:.4f} s"]
            for i, value in enumerate(np.ravel(values)):
                text.append(f"{self._component_label(key, i)}: {value:.6g}")
            ax_value.clear()
            ax_value.axis("off")
            ax_value.text(0.0, 1.0, "\n".join(text), va="top", family="monospace")

        def redraw(key):
            ax_plot.clear()
            values = np.asarray(arrays[key])
            if values.ndim == 1:
                line = ax_plot.plot(t, values, label=key)
            else:
                line = []
                for i in range(values.shape[1]):
                    line.extend(
                        ax_plot.plot(t, values[:, i], label=self._component_label(key, i))
                    )

            if key == "barrier":
                ax_plot.axhline(0.0, color="black", linestyle="--", linewidth=1, label="boundary")

            cursor = ax_plot.axvline(slider.val, color="black", linestyle="--", linewidth=1)
            lines[:] = [*line, cursor]
            ax_plot.set_title(key)
            ax_plot.set_xlabel("time [s]")
            ax_plot.set_ylabel(key)
            ax_plot.grid(True)
            ax_plot.legend(loc="best")
            update_value_panel(key, nearest_index(slider.val))
            fig.canvas.draw_idle()

        def on_slider_changed(value):
            state["index"] = nearest_index(value)
            if lines:
                lines[-1].set_xdata([value, value])
            update_value_panel(state["key"], state["index"])
            fig.canvas.draw_idle()

        def make_button_callback(key):
            def callback(_event):
                state["key"] = key
                redraw(key)

            return callback

        button_height = 0.055
        button_gap = 0.012
        top = 0.86
        for i, key in enumerate(plot_keys):
            bottom = top - i * (button_height + button_gap)
            ax_button = fig.add_axes([0.03, bottom, 0.16, button_height])
            button = Button(ax_button, key)
            button.on_clicked(make_button_callback(key))
            # Keep a reference so matplotlib does not garbage-collect callbacks.
            setattr(self, f"_gui_button_{key}", button)

        slider.on_changed(on_slider_changed)
        self._gui_slider = slider

        redraw(state["key"])
        plt.show()

        return fig

    @staticmethod
    def _configure_matplotlib_cache():
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    @staticmethod
    def _select_interactive_backend():
        """GUI 표시가 가능한 matplotlib backend를 선택한다.

        Notes:
            서버/SSH처럼 display가 없는 환경에서는 어떤 interactive backend도
            사용할 수 없다. 이 경우 저장된 PNG/CSV/NPZ를 사용해야 한다.
        """
        import matplotlib

        current_backend = matplotlib.get_backend().lower()
        if "agg" not in current_backend:
            return

        for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
            try:
                matplotlib.use(backend, force=True)
                return
            except Exception:
                continue

        raise RuntimeError(
            "No interactive matplotlib backend is available. "
            "Install tkinter/Qt support or run on a machine with a GUI display. "
            "You can still use plot(path=...) to save a PNG."
        )

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
        if key == "ee_yaw":
            return "ee_yaw [rad]"
        if key == "barrier" and index < len(self.barrier_labels):
            return self.barrier_labels[index]
        return f"{key}_{index}"

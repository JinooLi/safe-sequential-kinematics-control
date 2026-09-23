"""Static and interactive views of recorded arrays."""

import os
from pathlib import Path

import numpy as np

from ..logging.labels import component_label


def plot(arrays, barrier_labels, path=None, show=False):
    """기록된 데이터를 matplotlib으로 시각화한다.

    Args:
        path: 지정하면 figure를 이미지 파일로 저장한다. 예: "outputs/sim_plot.png".
        show: True이면 matplotlib 창을 띄운다.

    Returns:
        matplotlib Figure 객체.
    """
    _configure_matplotlib_cache()

    if "t" not in arrays:
        raise ValueError("No recorded data to plot.")

    plot_keys = [
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
                ax.plot(t, values[:, i], label=component_label(key, i, barrier_labels))
        if key == "barrier":
            ax.axhline(
                0.0, color="black", linestyle="--", linewidth=1, label="boundary"
            )
        ax.set_ylabel(key)
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    if show:
        plt.show()

    return fig


def show_gui(arrays, barrier_labels):
    """기록된 데이터를 matplotlib GUI로 탐색한다.

    왼쪽 버튼으로 표시할 데이터 그룹(q, dq, ee, u 등)을 바꿀 수 있고,
    아래 slider로 시간 위치를 움직이면 현재 시각의 값을 오른쪽 패널에서
    확인할 수 있다.

    Usage:
        recorder.show_gui()

        # 저장된 데이터를 나중에 보고 싶을 때
        Recorder.from_npz("outputs/sim_data.npz").show_gui()
    """
    _configure_matplotlib_cache()
    _select_interactive_backend()
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

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
            text.append(f"{component_label(key, i, barrier_labels)}: {value:.6g}")
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
                    ax_plot.plot(
                        t, values[:, i], label=component_label(key, i, barrier_labels)
                    )
                )

        if key == "barrier":
            ax_plot.axhline(
                0.0, color="black", linestyle="--", linewidth=1, label="boundary"
            )

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
        setattr(fig, f"_gui_button_{key}", button)

    slider.on_changed(on_slider_changed)
    fig._gui_slider = slider

    redraw(state["key"])
    plt.show()

    return fig


def _configure_matplotlib_cache():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


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

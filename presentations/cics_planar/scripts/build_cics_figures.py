"""저장된 Planar 실험 자료로 CICS 발표용 벡터 그래프 생성.

재실험이나 궤적 수정 없이 제출 원고의 가중치 표기로 CLF를 복원한다.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cics-planar-mpl")
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence")
args = parser.parse_args()
DATA = args.evidence_dir.resolve()
FIG = ROOT / "Fig"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{kotex}\usepackage[T1]{fontenc}\usepackage{lmodern}\renewcommand{\familydefault}{\sfdefault}",
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "text.color": "#000000", "axes.labelcolor": "#000000",
})


def read(name):
    return json.loads((DATA / name).read_text())


def save(fig, name):
    # bbox_inches="tight"를 사용하지 않아 지정한 발표 슬라이드 크기를 유지.
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    subprocess.run(["pdftoppm", "-r", "200", "-singlefile", "-png",
                    str(FIG / f"{name}.pdf"), str(FIG / name)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def fk(q):
    angles = np.cumsum(q, axis=-1)
    return np.cumsum(np.stack((np.cos(angles), np.sin(angles)), axis=-1), axis=-2)


report = read("case030_report.json")
case = report["original_case"]
reconstruction = report["lyapunov_reconstruction"]
with np.load(DATA / "case_030_sequential.npz", allow_pickle=False) as sq, \
     np.load(DATA / "case_030_direct.npz", allow_pickle=False) as di:
    t = sq["t"]
    active = sq["active_link"].astype(int)
    np.testing.assert_array_equal(t, di["t"])
    np.testing.assert_array_equal(sq["q"][0], di["q"][0])
    targets = np.asarray(reconstruction["targets"])
    np.testing.assert_allclose(fk(reconstruction["selected_ik_q"]), targets, atol=1e-12)
    points = fk(sq["q"])
    np.testing.assert_allclose(points[:, -1], sq["final_xy"], atol=1e-12)

    # 원기록 V=10*(||e_p||²+0.5*(1-cos(e_psi)))를 원고의
    # V=w_v*||e_p||²+w_psi*(1-cos(e_psi))로 정확히 재표기.
    w_v = 2 * reconstruction["clf_weight"]
    w_psi = w_v * reconstruction["final_yaw_weight"]
    clf = w_v * np.sum((points - targets) ** 2, axis=-1)
    clf[:, -1] += w_psi * (1 - np.cos(sq["final_yaw"] - case["target_yaw"]))
    direct = w_v * np.sum((di["final_xy"] - case["target"]) ** 2, axis=1)
    direct += w_psi * (1 - np.cos(di["final_yaw"] - case["target_yaw"]))
    active_error = float(np.max(np.abs(clf[np.arange(len(t)), active - 1] - sq["clf"])))
    direct_error = float(np.max(np.abs(direct - di["clf"])))
    np.testing.assert_allclose(clf[np.arange(len(t)), active - 1], sq["clf"], rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(direct, di["clf"], rtol=1e-11, atol=1e-10)
    changes = np.flatnonzero(np.diff(active) != 0) + 1
    np.testing.assert_allclose(t[changes], [2.128, 6.874], atol=1e-9)
    assert case["sequential"]["success"] and not case["direct"]["success"]

    colors = ["#0000FF", "#E68000", "#008000"]
    fig, axes = plt.subplots(1, 3, figsize=(118 / 25.4, 35 / 25.4))
    fig.subplots_adjust(left=.065, right=.97, bottom=.31, top=.83, wspace=.46)
    for j, ax in enumerate(axes):
        ax.plot(t, clf[:, j], color=colors[j], ls=(0, (3, 2)), lw=.8)
        ax.plot(t, np.where(active == j + 1, clf[:, j], np.nan), color=colors[j], lw=1.45)
        for transition in t[changes]:
            ax.axvline(transition, color="#888888", ls=":", lw=.65)
        if j < 2:
            start = 0 if j == 0 else t[changes[0]]
            ax.axvspan(start, t[changes[j]], color=colors[j], alpha=.07)
        else:
            ax.plot(t, direct, color="#FF0000", ls="-.", lw=1.15, label="Direct")
            ax.legend(loc="upper right", frameon=False, handlelength=.75,
                      handletextpad=.25, borderaxespad=0)
        ax.set_title(rf"$V_{j + 1}$", pad=2)
        ax.set_xlim(0, 35)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 20, 35])
        ax.grid(axis="y", alpha=.2, linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks([0, 20])
    axes[1].set_yticks([0, 40])
    axes[2].set_yticks([0, 100])
    fig.supxlabel("시간 (s)", y=.005, fontsize=10)
    save(fig, "cics_clf_time")

cases = read("planar_cases.json")
counts = [sum(c[method]["success"] for c in cases) for method in ["direct", "sequential"]]
assert len(cases) == 300 and counts == [216, 253]
rates = np.asarray(counts) / len(cases) * 100
fig, ax = plt.subplots(figsize=(118 / 25.4, 42 / 25.4))
fig.subplots_adjust(left=.19, right=.97, top=.92, bottom=.32)
ax.barh([0, 1], rates, height=.52, color=["#DADADA", "#C7D7FF"])
for index, (rate, count) in enumerate(zip(rates, counts)):
    ax.text(3, index, f"{rate:.2f}\\% ({count}/300)", va="center",
            color="#000000" if index == 0 else "#0000FF", fontsize=10)
ax.set_yticks([0, 1], ["Direct", "Sequential"])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel(r"성공률 (\%)", labelpad=3)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", alpha=.18)
ax.set_axisbelow(True)
save(fig, "cics_success")

input_names = ["case030_report.json", "case_030_sequential.npz", "case_030_direct.npz", "planar_cases.json"]
audit = {
    "figure_files": ["cics_clf_time.pdf", "cics_clf_time.png", "cics_success.pdf", "cics_success.png"],
    "font": "pdfLaTeX / ko.TeX / Latin Modern Sans; all text 10 pt at the native PDF dimensions",
    "figure_language": "Korean descriptive labels; Direct, Sequential, and mathematical symbols retained",
    "native_dimensions_mm": {"cics_clf_time": [118, 35], "cics_success": [118, 42]},
    "case_id": 30,
    "operation": "제출 원고의 가중치 표기에 맞춘 원기록 CLF의 정확한 복원; 재실험 없음",
    "paper_formula": "V_i=w_v*||p_i-p_i_star||^2 (i=1,2); V_3=w_v*||p_3-p_d||^2+w_psi*(1-cos(psi_3-psi_d))",
    "paper_weights": {"w_v": w_v, "w_psi": w_psi},
    "archived_convention": {"position_weight": w_v, "orientation_weight_inside_position_factor": reconstruction["final_yaw_weight"]},
    "weight_conversion_note": "원기록에서 0.5는 전체 w_v=10의 괄호 안에 위치하므로 원고의 독립 orientation 항 계수 w_psi는 5이다. 원고 본문에는 가중치 수치가 명시되지 않음.",
    "max_active_clf_absolute_error": active_error,
    "max_direct_clf_absolute_error": direct_error,
    "sample_count_each_method": int(len(t)),
    "transition_times_seconds": [float(x) for x in t[changes]],
    "sequential_completion_time_seconds": case["sequential"]["completion_time"],
    "direct_failure_reason": case["direct"]["failure_reason"],
    "no_new_simulation": True,
    "no_trajectory_smoothing_or_subsampling": True,
    "counts": {"episodes": 300, "direct_success": counts[0], "sequential_success": counts[1]},
    "rates_percent": {"direct": float(rates[0]), "sequential": float(rates[1])},
    "improvement_percentage_points": float(rates[1] - rates[0]),
    "source_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in input_names},
}
(ROOT / "evidence" / "cics_figures.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"files": audit["figure_files"], "max_clf_error": max(active_error, direct_error)}, ensure_ascii=False))

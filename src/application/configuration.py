"""Typed application configuration and cross-model geometry validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


@dataclass(frozen=True)
class ObstacleConfig:
    position: tuple[float, float]
    radius: float
    enabled: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool
    representation: str
    rdf_model_path: str
    link_lengths: tuple[float, ...]
    link_radius: float
    safe_distance: float
    alpha: float
    joint_limits: tuple[tuple[float, float] | None, ...]
    joint_limit_alpha: float
    velocity_limit: float
    clf_rate: float


@dataclass(frozen=True)
class CLBFConfig:
    gamma: float
    slack_alpha: float
    end_effector_radius: float
    barriers: tuple[CLBFBarrierConfig, ...]


@dataclass(frozen=True)
class CLBFBarrierConfig:
    """One end-effector obstacle barrier in hierarchical CLBF order.

    The first entry is the user-selected seed.  A later entry with
    ``auto_from_previous`` derives its epsilon/lambda/kappa from all previous
    entries and its own obstacle geometry, following the paper's recursion.
    """

    obstacle_index: int
    epsilon: float | None = None
    lambda_: float | None = None
    kappa: float | None = None
    auto_from_previous: bool = False


@dataclass(frozen=True)
class QPConfig:
    R: np.ndarray
    Q: np.ndarray
    d: float


@dataclass(frozen=True)
class SequentialEndEffectorConfig:
    """Settings for progressing the task point from link 1 to the final tip."""

    enabled: bool
    frame_names: tuple[str, ...]
    goal_tolerance: float
    velocity_tolerance: float
    barrier_tolerance: float


@dataclass(frozen=True)
class SequentialExperimentConfig:
    """All randomized sequential-control comparison settings."""

    cases: int
    workers: int
    seed: int
    duration: float
    goal_tolerance: float
    yaw_tolerance: float
    barrier_tolerance: float
    control_mode: str | None
    initial_q: tuple[float, ...]
    output_dir: str
    target_radius_range: tuple[float, float]
    target_angle_range: tuple[float, float]
    target_orientation_range: tuple[float, float]
    obstacle_position_range: tuple[float, float]
    obstacle_radius_range: tuple[float, float]
    initial_clearance_margin: float
    target_clearance_margin: float
    max_case_attempts: int
    max_obstacle_attempts: int


@dataclass(frozen=True)
class ControllerConfig:
    dof: int
    ee_target: np.ndarray
    ee_orientation: float
    control_mode: str
    obstacles: tuple[ObstacleConfig, ...]
    safety: SafetyConfig
    clbf: CLBFConfig
    qp: QPConfig
    sequential_end_effector: SequentialEndEffectorConfig
    sequential_experiment: SequentialExperimentConfig


def load_controller_config(path: str | Path) -> ControllerConfig:
    """Load controller.yaml once, validate its shape, and return typed data."""
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    safety_data = data["safety"]
    dof = int(data["dof"])
    clbf_data = data["clbf"]
    qp_data = data["QP_ctrl"]
    sequential_data = data.get("sequential_end_effector", {})
    experiment_data = data.get("sequential_experiment", {})
    obstacles = tuple(
        ObstacleConfig(
            tuple(map(float, entry["pos"])),
            float(entry["radius"]),
            bool(entry.get("enabled", True)),
        )
        for entry in data.get("obstacle", [])
    )
    joint_limits = tuple(
        None if limit is None else tuple(map(float, limit))
        for limit in safety_data.get("joint_limits", [None] * dof)
    )
    safety = SafetyConfig(
        enabled=bool(safety_data["enabled"]),
        representation=str(safety_data["representation"]),
        rdf_model_path=str(safety_data["rdf_model_path"]),
        link_lengths=tuple(map(float, safety_data["link_lengths"])),
        link_radius=float(safety_data["link_radius"]),
        safe_distance=float(safety_data["safe_distance"]),
        alpha=float(safety_data["alpha"]),
        joint_limits=joint_limits,
        joint_limit_alpha=float(safety_data.get("joint_limit_alpha", safety_data["alpha"])),
        velocity_limit=float(safety_data["velocity_limit"]),
        clf_rate=float(safety_data["clf_rate"]),
    )
    clbf = CLBFConfig(
        gamma=float(clbf_data["gamma"]),
        slack_alpha=float(clbf_data.get("slack_alpha", 1.0)),
        end_effector_radius=float(clbf_data["end_effector_radius"]),
        barriers=tuple(
            CLBFBarrierConfig(
                obstacle_index=int(barrier["obstacle_index"]),
                epsilon=(float(barrier["epsilon"]) if "epsilon" in barrier else None),
                lambda_=(float(barrier["lambda"]) if "lambda" in barrier else None),
                kappa=(float(barrier["kappa"]) if "kappa" in barrier else None),
                auto_from_previous=bool(barrier.get("auto_from_previous", False)),
            )
            for barrier in clbf_data["barriers"]
        ),
    )
    config = ControllerConfig(
        dof=dof,
        ee_target=np.asarray(data["ee_obj"], dtype=float),
        ee_orientation=float(data.get("ee_orientation", 0.0)),
        control_mode=str(data.get("control_mode", "clf")),
        obstacles=obstacles,
        safety=safety,
        clbf=clbf,
        qp=QPConfig(
            np.asarray(qp_data["R"], dtype=float),
            np.asarray(qp_data["Q"], dtype=float),
            float(qp_data["d"]),
        ),
        sequential_end_effector=SequentialEndEffectorConfig(
            enabled=bool(sequential_data.get("enabled", False)),
            frame_names=tuple(map(str, sequential_data.get("frame_names", ()))),
            goal_tolerance=float(sequential_data.get("goal_tolerance", 0.03)),
            velocity_tolerance=float(sequential_data.get("velocity_tolerance", 1e-3)),
            barrier_tolerance=float(sequential_data.get("barrier_tolerance", 1e-3)),
        ),
        sequential_experiment=SequentialExperimentConfig(
            cases=int(experiment_data.get("cases", 100)),
            workers=int(experiment_data.get("workers", 1)),
            seed=int(experiment_data.get("seed", 0)),
            duration=float(experiment_data.get("duration", 25.0)),
            goal_tolerance=float(experiment_data.get("goal_tolerance", 0.10)),
            yaw_tolerance=float(experiment_data.get("yaw_tolerance", 0.10)),
            barrier_tolerance=float(experiment_data.get("barrier_tolerance", -1e-3)),
            control_mode=(
                str(experiment_data["control_mode"])
                if experiment_data.get("control_mode") is not None
                else None
            ),
            initial_q=tuple(map(float, experiment_data.get("initial_q", [0.0] * int(data["dof"])))),
            output_dir=str(experiment_data.get("output_dir", "outputs/sequential_comparison")),
            target_radius_range=tuple(map(float, experiment_data.get("target_radius_range", (0.6, 2.7)))),
            target_angle_range=tuple(map(float, experiment_data.get("target_angle_range", (-np.pi, np.pi)))),
            target_orientation_range=tuple(map(float, experiment_data.get("target_orientation_range", (-np.pi, np.pi)))),
            obstacle_position_range=tuple(map(float, experiment_data.get("obstacle_position_range", (-2.2, 2.2)))),
            obstacle_radius_range=tuple(map(float, experiment_data.get("obstacle_radius_range", (0.2, 0.35)))),
            initial_clearance_margin=float(experiment_data.get("initial_clearance_margin", 0.03)),
            target_clearance_margin=float(experiment_data.get("target_clearance_margin", 0.10)),
            max_case_attempts=int(experiment_data.get("max_case_attempts", 10_000)),
            max_obstacle_attempts=int(experiment_data.get("max_obstacle_attempts", 1_000)),
        ),
    )
    if config.dof != len(config.safety.link_lengths):
        raise ValueError("dof and safety.link_lengths must have the same length")
    if len(config.safety.joint_limits) != config.dof:
        raise ValueError("safety.joint_limits must contain one entry per dof")
    for index, limit in enumerate(config.safety.joint_limits, start=1):
        if limit is None:
            continue
        if len(limit) != 2 or not np.all(np.isfinite(limit)) or limit[0] >= limit[1]:
            raise ValueError(
                f"safety.joint_limits[{index}] must be null or an increasing finite [lower, upper] pair"
            )
    if config.safety.joint_limit_alpha <= 0.0:
        raise ValueError("safety.joint_limit_alpha must be positive")
    if config.ee_target.shape != (2,):
        raise ValueError("ee_obj must contain exactly [x, y]")
    if not np.isfinite(config.ee_orientation):
        raise ValueError("ee_orientation must be finite")
    if config.control_mode not in {"clf", "clbf"}:
        raise ValueError("control_mode must be 'clf' or 'clbf'")
    if config.qp.R.shape != (config.dof, config.dof) or config.qp.Q.shape != (1, 1):
        raise ValueError("QP_ctrl must contain a dof-by-dof R and 1-by-1 Q")
    if not 0.0 < config.qp.d <= 1.0:
        raise ValueError("QP_ctrl.d must satisfy 0 < d <= 1")
    sequence = config.sequential_end_effector
    if sequence.enabled and len(sequence.frame_names) != config.dof:
        raise ValueError(
            "sequential_end_effector.frame_names must contain one frame for each link"
        )
    if len(set(sequence.frame_names)) != len(sequence.frame_names):
        raise ValueError("sequential_end_effector.frame_names must not repeat a frame")
    if (
        sequence.goal_tolerance <= 0.0
        or sequence.velocity_tolerance < 0.0
        or sequence.barrier_tolerance < 0.0
    ):
        raise ValueError("sequential end-effector tolerances must be non-negative")
    experiment = config.sequential_experiment
    if len(experiment.initial_q) != config.dof or not np.all(np.isfinite(experiment.initial_q)):
        raise ValueError("sequential_experiment.initial_q must contain one finite value per dof")
    if (
        experiment.cases < 1
        or experiment.workers < 1
        or experiment.duration <= 0.0
        or experiment.goal_tolerance <= 0.0
        or experiment.yaw_tolerance <= 0.0
        or experiment.initial_clearance_margin < 0.0
        or experiment.target_clearance_margin < 0.0
        or experiment.max_case_attempts < 1
        or experiment.max_obstacle_attempts < 1
    ):
        raise ValueError("sequential experiment counts, duration, and tolerances are invalid")
    for name, value_range in (
        ("target_radius_range", experiment.target_radius_range),
        ("target_angle_range", experiment.target_angle_range),
        ("target_orientation_range", experiment.target_orientation_range),
        ("obstacle_position_range", experiment.obstacle_position_range),
        ("obstacle_radius_range", experiment.obstacle_radius_range),
    ):
        if len(value_range) != 2 or not value_range[0] < value_range[1]:
            raise ValueError(f"sequential_experiment.{name} must be an increasing two-value range")
    if experiment.target_radius_range[0] <= 0.0 or experiment.obstacle_radius_range[0] <= 0.0:
        raise ValueError("sequential experiment radii must be positive")
    if experiment.control_mode not in {None, "clf", "clbf"}:
        raise ValueError("sequential_experiment.control_mode must be clf, clbf, or null")
    if config.safety.representation not in {"capsule", "bernstein"}:
        raise ValueError("safety representation must be 'capsule' or 'bernstein'")
    if config.control_mode == "clbf" and not config.safety.enabled:
        raise ValueError("control_mode 'clbf' requires safety.enabled")
    if config.clbf.gamma <= 0.0:
        raise ValueError("clbf gamma must be positive")
    if config.clbf.slack_alpha <= 0.0:
        raise ValueError("clbf slack_alpha must be positive")
    if config.clbf.end_effector_radius <= 0.0:
        raise ValueError("clbf end_effector_radius must be positive")
    active_obstacle_indices = {
        index for index, obstacle in enumerate(config.obstacles, start=1) if obstacle.enabled
    }
    configured_indices = tuple(barrier.obstacle_index for barrier in config.clbf.barriers)
    if config.control_mode == "clbf" and set(configured_indices) != active_obstacle_indices:
        raise ValueError("clbf.barriers must contain each active obstacle exactly once")
    if len(configured_indices) != len(set(configured_indices)):
        raise ValueError("clbf.barriers must not repeat an obstacle_index")
    if config.control_mode == "clbf" and not config.clbf.barriers:
        raise ValueError("clbf.barriers must not be empty in clbf mode")
    for index, barrier in enumerate(config.clbf.barriers):
        values = (barrier.epsilon, barrier.lambda_, barrier.kappa)
        if barrier.auto_from_previous:
            if index == 0:
                raise ValueError("the first CLBF barrier cannot be auto_from_previous")
            if any(value is not None for value in values):
                raise ValueError("an automatic CLBF barrier must not set epsilon, lambda, or kappa")
        elif any(value is None for value in values):
            raise ValueError("a manual CLBF barrier needs epsilon, lambda, and kappa")
        elif barrier.epsilon <= 0.0 or barrier.lambda_ <= 0.0 or not np.isfinite(barrier.kappa):
            raise ValueError("manual CLBF epsilon/lambda must be positive and kappa finite")
    return config


def _numbered_links(elements, extract):
    result = {}
    for element in elements:
        name = element.get("name", "")
        if not name.startswith("link") or not name[4:].isdigit():
            continue
        result[int(name[4:])] = extract(element)
    return [result[index] for index in sorted(result)]


def _urdf_link_geometry(path: str | Path):
    root = ET.parse(path).getroot()

    def extract(link):
        cylinder = link.find("./collision/geometry/cylinder")
        if cylinder is None:
            raise ValueError(f"URDF {link.get('name')} must contain a collision cylinder")
        return float(cylinder.attrib["length"]), float(cylinder.attrib["radius"])

    return _numbered_links(root.findall("./link"), extract)


def _mujoco_link_geometry(path: str | Path):
    root = ET.parse(path).getroot()

    def extract(body):
        geom = body.find("./geom[@type='cylinder']")
        if geom is None:
            raise ValueError(f"MuJoCo {body.get('name')} must contain a cylinder geom")
        size = tuple(map(float, geom.attrib["size"].split()))
        if len(size) != 2:
            raise ValueError(f"MuJoCo {body.get('name')} cylinder must have radius and half-length")
        return 2.0 * size[1], size[0]

    return _numbered_links(root.findall(".//body"), extract)


def _mujoco_obstacle_geometry(path: str | Path):
    root = ET.parse(path).getroot()
    result = {}
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if not name.startswith("obstacle_geom") or not name[13:].isdigit():
            continue
        if geom.get("type") != "cylinder":
            raise ValueError(f"MuJoCo {name} must be a cylinder")
        position = tuple(map(float, geom.attrib["pos"].split()[:2]))
        radius = float(geom.attrib["size"].split()[0])
        result[int(name[13:])] = (position, radius)
    return [result[index] for index in sorted(result)]


def _mujoco_end_effector_radius(path: str | Path):
    root = ET.parse(path).getroot()
    geom = root.find(".//geom[@name='ee_geom']")
    if geom is None or geom.get("type") != "sphere":
        raise ValueError("MuJoCo must contain a sphere geom named ee_geom")
    return float(geom.attrib["size"].split()[0])


def validate_geometry_consistency(
    config: ControllerConfig,
    urdf_path: str | Path,
    mujoco_path: str | Path,
    tolerance: float = 1e-9,
):
    """Fail fast when controller, URDF, and MuJoCo geometry disagree.

    The controller's capsule is conservative in the plane, but it must use
    the same axial length and radius as the collision cylinders in both robot
    descriptions.  Circular obstacle radius/position are checked as well.
    """
    expected_links = list(zip(config.safety.link_lengths, [config.safety.link_radius] * config.dof))
    sources = {
        "controller.yaml": expected_links,
        "URDF": _urdf_link_geometry(urdf_path),
        "MuJoCo": _mujoco_link_geometry(mujoco_path),
    }
    reference_name, reference = next(iter(sources.items()))
    for source_name, geometry in sources.items():
        if len(geometry) != config.dof:
            raise ValueError(f"{source_name} has {len(geometry)} links; expected {config.dof}")
        for index, (actual, expected) in enumerate(zip(geometry, reference), start=1):
            if not np.allclose(actual, expected, atol=tolerance, rtol=0.0):
                raise ValueError(
                    f"geometry mismatch for link{index}: {source_name}={actual}, "
                    f"{reference_name}={expected}"
                )

    mujoco_obstacles = _mujoco_obstacle_geometry(mujoco_path)
    if len(mujoco_obstacles) != len(config.obstacles):
        raise ValueError("controller.yaml and MuJoCo have different obstacle counts")
    for index, (mujoco_obstacle, configured) in enumerate(
        zip(mujoco_obstacles, config.obstacles), start=1
    ):
        actual = (*mujoco_obstacle[0], mujoco_obstacle[1])
        expected = (*configured.position, configured.radius)
        if not np.allclose(actual, expected, atol=tolerance, rtol=0.0):
            raise ValueError(
                f"geometry mismatch for obstacle{index}: MuJoCo={actual}, controller.yaml={expected}"
            )

    ee_radius = _mujoco_end_effector_radius(mujoco_path)
    if not np.isclose(ee_radius, config.clbf.end_effector_radius, atol=tolerance, rtol=0.0):
        raise ValueError(
            "geometry mismatch for end effector radius: "
            f"MuJoCo={ee_radius}, controller.yaml={config.clbf.end_effector_radius}"
        )

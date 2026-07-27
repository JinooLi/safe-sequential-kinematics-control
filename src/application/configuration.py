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


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool
    representation: str
    rdf_model_path: str
    link_lengths: tuple[float, ...]
    link_radius: float
    safe_distance: float
    alpha: float
    velocity_limit: float
    clf_rate: float


@dataclass(frozen=True)
class QPConfig:
    R: np.ndarray
    Q: np.ndarray


@dataclass(frozen=True)
class ControllerConfig:
    dof: int
    ee_target: np.ndarray
    obstacles: tuple[ObstacleConfig, ...]
    safety: SafetyConfig
    qp: QPConfig


def load_controller_config(path: str | Path) -> ControllerConfig:
    """Load controller.yaml once, validate its shape, and return typed data."""
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    safety_data = data["safety"]
    qp_data = data["QP_ctrl"]
    obstacles = tuple(
        ObstacleConfig(tuple(map(float, entry["pos"])), float(entry["radius"]))
        for entry in data.get("obstacle", [])
    )
    safety = SafetyConfig(
        enabled=bool(safety_data["enabled"]),
        representation=str(safety_data["representation"]),
        rdf_model_path=str(safety_data["rdf_model_path"]),
        link_lengths=tuple(map(float, safety_data["link_lengths"])),
        link_radius=float(safety_data["link_radius"]),
        safe_distance=float(safety_data["safe_distance"]),
        alpha=float(safety_data["alpha"]),
        velocity_limit=float(safety_data["velocity_limit"]),
        clf_rate=float(safety_data["clf_rate"]),
    )
    config = ControllerConfig(
        dof=int(data["dof"]),
        ee_target=np.asarray(data["ee_obj"], dtype=float),
        obstacles=obstacles,
        safety=safety,
        qp=QPConfig(np.asarray(qp_data["R"], dtype=float), np.asarray(qp_data["Q"], dtype=float)),
    )
    if config.dof != len(config.safety.link_lengths):
        raise ValueError("dof and safety.link_lengths must have the same length")
    if config.ee_target.shape != (2,):
        raise ValueError("ee_obj must contain exactly [x, y]")
    if config.qp.R.shape != (config.dof, config.dof) or config.qp.Q.shape != (1, 1):
        raise ValueError("QP_ctrl must contain a dof-by-dof R and 1-by-1 Q")
    if config.safety.representation not in {"capsule", "bernstein"}:
        raise ValueError("safety representation must be 'capsule' or 'bernstein'")
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

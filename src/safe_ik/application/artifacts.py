"""Run directories and provenance shared by Panda and planar experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def create_run_directory(parent: str | Path, name: str | None = None) -> Path:
    if name is None:
        name = (
            datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ_")
            + uuid4().hex[:8]
        )
    if Path(name).name != name or name in {".", "..", ""}:
        raise ValueError("run name must be a single directory name")
    path = Path(parent).resolve() / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_resources(path: Path) -> set[Path]:
    """Include referenced MJCF meshes/includes as well as the model itself."""
    found: set[Path] = set()

    def visit(current):
        current = current.resolve()
        if current in found:
            return
        found.add(current)
        if current.suffix not in {".xml", ".urdf"}:
            return
        root = ET.parse(current).getroot()
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        for element in root.findall(".//mesh[@file]"):
            found.add((current.parent / meshdir / element.attrib["file"]).resolve())
        for element in root.findall(".//include[@file]"):
            visit(current.parent / element.attrib["file"])

    visit(path)
    return found


def save_run_provenance(
    directory: Path,
    *,
    config_path: Path,
    config,
    root: Path,
    resources: list[Path],
    effective_yaml: dict | None = None,
) -> None:
    """Save reloadable YAML, resolved values, source/model hashes and Git state."""
    directory = directory.resolve()
    config_path = config_path.resolve()
    root = root.resolve()
    raw = (
        yaml.safe_load(config_path.read_text())
        if effective_yaml is None
        else effective_yaml
    )
    # Resource paths in the archived input must not depend on its new location.
    if "robot" in raw:
        raw["robot"]["model_path"] = str(config.model_path)
    if "safety" in raw and "rdf_model_path" in raw["safety"]:
        raw["safety"]["rdf_model_path"] = str(config.safety.rdf_model_path)
    if getattr(config.safety, "self_collision_model", None) == "rdf" or (
        getattr(config, "ik_self_collision_model", None) == "rdf"
        and config.trajectory.mode != "direct"
    ):
        raw["safety"].setdefault("rdf", {})["artifact"] = str(
            config.safety.rdf.artifact
        )
        if config.safety.rdf.artifact.is_file():
            resources = [*resources, config.safety.rdf.artifact]
    (directory / "config.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    (directory / "resolved_config.json").write_text(
        json.dumps(json_value(config), indent=2), encoding="utf-8"
    )

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    git_status = git("status", "--porcelain=v1", "--untracked-files=all")
    paths = set()
    for path in resources:
        paths.update(_model_resources(Path(path)))
    package_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(path.relative_to(package_root)): _sha256(path)
        for path in sorted(package_root.rglob("*.py"))
    }
    versions = {}
    for package in (
        "safe-inv-clf-robot-arm",
        "numpy",
        "scipy",
        "mujoco",
        "pin",
        "osqp",
        "pyyaml",
        "matplotlib",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "command": sys.argv,
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git_status) if git_status is not None else None,
            "status": git_status,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "resource_sha256": {str(path): _sha256(path) for path in sorted(paths)},
        "source_sha256": source_hashes,
        "lock_sha256": (
            _sha256(root / "uv.lock") if (root / "uv.lock").is_file() else None
        ),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

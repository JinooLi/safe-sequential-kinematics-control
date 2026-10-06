"""Read experiment YAML, merge controller settings, and preserve input snapshots."""

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..domain.panda_config import PandaControllerConfig
from ..domain.panda_sampling_config import PandaSamplingConfig
from .artifacts import json_value
from .panda_configuration import parse_panda_config


def _mapping(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
    return value


def _batch(value, name):
    _mapping(value, ("seeds", "cases_per_seed"), name)
    seeds, count = value.get("seeds"), value.get("cases_per_seed")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
        or type(count) is not int
        or count < 1
    ):
        raise ValueError(
            f"{name} needs unique nonnegative seeds and positive cases_per_seed"
        )
    return deepcopy(value)


def _merge(base, override, name="controller_overrides"):
    _mapping(override, base, name)
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(base[key], dict):
            result[key] = _merge(base[key], value, f"{name}.{key}")
        else:
            result[key] = deepcopy(value)
    return result


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PandaExperimentSettings:
    path: Path
    source_yaml: str
    controller_source_yaml: str
    config_path: Path
    config: PandaControllerConfig
    controller_yaml: dict
    main: dict
    sampling: PandaSamplingConfig
    subdivisions: int
    output_root: Path
    output: Path | None

    def identity(self):
        # Check live inputs before recording the experiment identity or snapshots.
        if self.path.read_bytes().decode("utf-8") != self.source_yaml:
            raise ValueError("experiment YAML changed since loading")
        if self.config_path.read_bytes().decode("utf-8") != self.controller_source_yaml:
            raise ValueError("controller YAML changed since loading")
        return {
            "experiment_sha256": _hash(self.source_yaml),
            "controller_sha256": _hash(self.controller_source_yaml),
            "main": self.main,
            "sampling": json_value(self.sampling),
            "subdivisions": self.subdivisions,
        }

    def resolved(self, output):
        return {
            "controller_config": "config.yaml",
            "rdf_device": self.config.safety.rdf.runtime_device,
            "output_root": str(self.output_root),
            "output": str(Path(output).resolve()),
            "main": self.main,
            "sampling": json_value(self.sampling),
            "validation": {"subdivisions": self.subdivisions},
            "controller_overrides": {},
        }

    def snapshot(self, directory):
        """Write exact input copies and runnable resolved YAML; return their hashes."""
        self.identity()
        directory = Path(directory)
        raw = deepcopy(self.controller_yaml)
        raw["robot"]["model_path"] = str(self.config.model_path)
        raw["safety"]["rdf"]["artifact"] = str(self.config.safety.rdf.artifact)
        contents = {
            "experiment.yaml": self.source_yaml,
            "controller_source.yaml": self.controller_source_yaml,
            "experiment.resolved.yaml": yaml.safe_dump(
                self.resolved(directory), sort_keys=False
            ),
            "config.yaml": yaml.safe_dump(raw, sort_keys=False),
        }
        for name, text in contents.items():
            path = directory / name
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(text.encode("utf-8"))
            temporary.replace(path)
        return {name: _hash(text) for name, text in contents.items()}


def load_panda_experiment(
    path,
    *,
    controller_path=None,
    device=None,
    duration=None,
    seeds=None,
    cases_per_seed=None,
    output=None,
):
    path = Path(path).resolve()
    source = path.read_bytes().decode("utf-8")
    data = _mapping(
        yaml.safe_load(source),
        (
            "controller_config",
            "rdf_device",
            "output_root",
            "output",
            "main",
            "sampling",
            "validation",
            "controller_overrides",
        ),
        "experiment",
    )
    config_path = (
        Path(controller_path).resolve()
        if controller_path is not None
        else (path.parent / data["controller_config"]).resolve()
    )
    controller_source = config_path.read_bytes().decode("utf-8")
    raw = yaml.safe_load(controller_source)
    base = parse_panda_config(raw, config_path)
    # Include supported optional controller keys when the base YAML omitted them.
    for section, defaults in (
        ("trajectory", base.trajectory),
        ("qp", base.qp),
        ("safe_ik", base.safe_ik),
    ):
        for key, value in json_value(defaults).items():
            raw[section].setdefault(key, value)
    raw = _merge(raw, data.get("controller_overrides", {}))
    device = device if device is not None else data.get("rdf_device", "cuda")
    if device not in ("cpu", "cuda"):
        raise ValueError("rdf_device must be cpu or cuda")
    raw["safety"].setdefault("rdf", {})["runtime_device"] = device
    if duration is not None:
        raw["trajectory"]["duration"] = duration
    config = parse_panda_config(raw, config_path)
    steps = config.trajectory.duration / config.trajectory.integration_period
    if not np.isclose(round(steps), steps):
        raise ValueError("duration must contain an integer number of control periods")
    main = _batch(
        data.get(
            "main", {"seeds": [2332, 2333, 2334, 2335, 2336], "cases_per_seed": 60}
        ),
        "main",
    )
    if seeds is not None:
        main["seeds"] = seeds
    if cases_per_seed is not None:
        main["cases_per_seed"] = cases_per_seed
    _batch(main, "main")
    sampling_data = _mapping(
        data.get("sampling", {}), PandaSamplingConfig.__dataclass_fields__, "sampling"
    )
    sampling = PandaSamplingConfig(**sampling_data)
    validation = _mapping(data.get("validation", {}), ("subdivisions",), "validation")
    subdivisions = validation.get("subdivisions", 5)
    if type(subdivisions) is not int or subdivisions < 1:
        raise ValueError("validation.subdivisions must be a positive integer")
    if sampling.max_proposals_per_seed < main["cases_per_seed"]:
        raise ValueError("sampling proposal budget is smaller than cases_per_seed")
    output_root = (
        path.parent / data.get("output_root", "../../outputs/panda_comparison")
    ).resolve()
    destination = (
        Path(output).resolve()
        if output is not None
        else (path.parent / data["output"]).resolve() if data.get("output") else None
    )
    return PandaExperimentSettings(
        path,
        source,
        controller_source,
        config_path,
        config,
        raw,
        main,
        sampling,
        subdivisions,
        output_root,
        destination,
    )

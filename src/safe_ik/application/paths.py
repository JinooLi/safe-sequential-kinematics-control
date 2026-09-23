"""Default resources for an editable checkout; explicit paths remain supported."""

from pathlib import Path


def project_root() -> Path:
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "configs").is_dir() and (checkout / "pyproject.toml").is_file():
        return checkout
    return Path.cwd()


def default_resource(relative: str) -> str:
    return str(project_root() / relative)

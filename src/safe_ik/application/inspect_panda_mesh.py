"""Export original Panda visual meshes with topology defects highlighted."""

import argparse
from pathlib import Path

from ..adapters.panda_rdf_geometry import extract_visual_geometry
from ..visualization.mesh_inspection import write_inspection_html
from .panda_configuration import load_panda_config
from .panda_rdf_artifact import read_reconstructed_meshes
from .paths import default_resource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_resource("configs/panda.yaml"))
    parser.add_argument("--output", default="outputs/panda_mesh_inspection/index.html")
    parser.add_argument(
        "--artifact",
        help="include reconstructed reference from an RDF candidate (may be incomplete/failed)",
    )
    args = parser.parse_args()
    config = load_panda_config(args.config)
    geometry = extract_visual_geometry(config.model_path)
    reconstructed, comparison = {}, {}
    if args.artifact:
        reconstructed, metadata = read_reconstructed_meshes(args.artifact)
        if metadata["geometry_fingerprint"] != geometry.fingerprint:
            parser.error("artifact and original visual geometry mismatch")
        comparison = metadata["reconstruction"]
    report = write_inspection_html(geometry, args.output, reconstructed, comparison)
    for name, link in report["links"].items():
        print(f"{name}: {link['edge_counts']}, regions={len(link['regions'])}")
    print(f"Saved: {Path(args.output).resolve()}")
    print(
        "Inspection only: source meshes, training models and safety settings unchanged."
    )


if __name__ == "__main__":
    main()

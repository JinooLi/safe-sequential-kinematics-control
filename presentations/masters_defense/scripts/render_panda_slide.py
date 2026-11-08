"""Render the archived Panda scene with MuJoCo meshes, without dynamics.

Uses the original repository model and its mesh assets read-only. Run with
the repository's .venv Python. --repo-root also supports an exported deck
whose parent directories no longer contain assets/robots/panda.xml.
"""

from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import os
import xml.etree.ElementTree as ET

# Set the rendering backend before importing MuJoCo or OpenGL. PyOpenGL
# 3.1.0 omits the opaque EGLDeviceEXT handle typedef used by current MuJoCo.
# Add only that standard pointer typedef in this process; no package edits.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import OpenGL
from OpenGL import EGL
EGL_TYPEDEF_COMPAT = not hasattr(EGL, "EGLDeviceEXT")
if EGL_TYPEDEF_COMPAT:
    EGL.EGLDeviceEXT = ctypes.c_void_p

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = Path("/home/jinoo/prj/safe-inv-clbf-robot-arm")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def locate_repository(explicit):
    candidates = [Path(explicit)] if explicit else [*ROOT.parents, DEFAULT_REPOSITORY]
    for candidate in candidates:
        if (candidate / "assets/robots/panda.xml").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Pass --repo-root pointing to safe-inv-clbf-robot-arm")


def build_model(repo, record, width, height):
    source = repo / "assets/robots/panda.xml"
    source_tree = ET.parse(source)
    tree = source_tree.getroot()
    compiler = tree.find("compiler")
    mesh_root = (source.parent / compiler.attrib["meshdir"]).resolve()
    compiler.set("meshdir", str(mesh_root))
    asset = tree.find("asset")
    meshes = [(mesh_root / element.attrib["file"]).resolve()
              for element in asset.findall("mesh")]
    assert all(path.is_file() for path in meshes)

    # Render-only materials/lighting. Kinematic chains and all robot mesh
    # geometry remain the original model; the stored obstacle list replaces
    # the source model's two demonstration obstacles.
    ET.SubElement(asset, "texture", {
        "name": "slide_sky", "type": "skybox", "builtin": "gradient",
        "rgb1": "1 1 1", "rgb2": "1 1 1", "width": "512", "height": "3072",
    })
    visual = ET.SubElement(tree, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(width), "offheight": str(height), "fovy": "35"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096", "offsamples": "8"})
    ET.SubElement(visual, "headlight", {
        "ambient": "0.15 0.15 0.15", "diffuse": "0.4 0.4 0.4", "specular": "0.12 0.12 0.12",
    })
    world = tree.find("worldbody")
    for element in list(world):
        if element.tag == "light":
            world.remove(element)
        elif element.tag == "geom":
            if element.attrib.get("name") == "floor":
                # Hide the visual plane to keep a seamless white backdrop.
                element.set("rgba", "1 1 1 0")
                element.set("size", "5 5 0.05")
            else:
                # Remove example obstacles and base coordinate decorations.
                world.remove(element)
        elif element.tag == "site":
            # A renderer-owned goal marker and triad are added below.
            world.remove(element)
    ET.SubElement(world, "light", {
        "name": "slide_key", "pos": "1 -2 3", "dir": "-0.25 0.5 -1",
        "directional": "true", "diffuse": "0.5 0.5 0.5", "ambient": "0.04 0.04 0.04",
        "specular": "0.15 0.15 0.15", "castshadow": "false",
    })
    ET.SubElement(world, "light", {
        "name": "slide_fill", "pos": "-2 1 2", "dir": "0.6 -0.25 -1",
        "directional": "true", "diffuse": "0.15 0.15 0.15", "ambient": "0 0 0",
        "specular": "0.1 0.1 0.1", "castshadow": "false",
    })
    for index, obstacle in enumerate(record["obstacles"]):
        ET.SubElement(world, "geom", {
            "name": f"slide_obstacle_{index}", "type": "sphere",
            "pos": numbers(obstacle[:3]), "size": numbers(obstacle[3:]),
            "rgba": "0.65 0.23 0.16 0.17", "contype": "0", "conaffinity": "0",
        })
    for site in tree.iter("site"):
        site.set("rgba", "0 0 0 0")
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    return model, source, meshes, xml


def add_target(scene, target, rotation):
    sphere = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(sphere, mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.asarray([.028, .028, .028]), target, np.eye(3).ravel(),
                       np.asarray([0., .62, .45, 1.], dtype=np.float32))
    scene.ngeom += 1
    for index, rgba in enumerate([[.82, .14, .13, 1], [0., .62, .45, 1], [.13, .33, .65, 1]]):
        arrow = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, .008,
                            target, target + .15 * rotation[:, index])
        scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--azimuth", type=float, default=45)
    parser.add_argument("--elevation", type=float, default=-25)
    parser.add_argument("--distance", type=float, default=2.7)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    repo = locate_repository(args.repo_root)
    record_path = ROOT / "evidence/panda_example.json"
    record = json.loads(record_path.read_text())
    width, height = 1800, 1350
    model, source, mesh_files, xml = build_model(repo, record, width, height)
    assert sha(source) == record["model_sha256"], "Archived model differs from current model"
    data = mujoco.MjData(model)
    assert model.nq == 7
    data.qpos[:] = record["initial_q"]
    mujoco.mj_forward(model, data)
    positions = []
    for name in record["frame_names"]:
        try:
            position = data.body(name).xpos.copy()
        except KeyError:
            position = data.site(name).xpos.copy()
        positions.append(position)
    residual = float(np.max(np.abs(np.asarray(positions) - np.asarray(record["initial_frame_positions"]))))
    np.testing.assert_allclose(positions, record["initial_frame_positions"], atol=1e-12)
    for index, obstacle in enumerate(record["obstacles"]):
        geom = model.geom(f"slide_obstacle_{index}")
        np.testing.assert_allclose(geom.pos, obstacle[:3], atol=1e-12)
        np.testing.assert_allclose(geom.size[0], obstacle[3], atol=1e-12)
    target, rotation = np.asarray(record["target_position"]), np.asarray(record["target_rotation"])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.02, .02, .51]
    camera.distance = args.distance
    camera.elevation = args.elevation
    camera.azimuth = args.azimuth
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # Original collision meshes are not visual surfaces.
    option.sitegroup[:] = 0
    output = ROOT / "Fig/panda_mujoco.png"
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        def render(path, azimuth):
            camera.azimuth = azimuth
            renderer.update_scene(data, camera=camera, scene_option=option)
            add_target(renderer.scene, target, rotation)
            pixels = renderer.render()
            Image.fromarray(pixels).save(path)
            visible_y, visible_x = np.nonzero(np.min(pixels, axis=2) < 248)
            return [int(visible_x.min()), int(visible_y.min()),
                    int(visible_x.max()), int(visible_y.max())]
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            for azimuth in [-45, 45, 135]:
                render(args.preview_dir / f"azimuth_{azimuth}.png", azimuth)
        visible_bbox = render(output, args.azimuth)
    edge_margins = [visible_bbox[0], visible_bbox[1],
                    width - 1 - visible_bbox[2], height - 1 - visible_bbox[3]]
    assert min(edge_margins) >= 30, "Visible scene is too close to the image edge"
    assert data.time == 0 and np.all(data.qvel == 0)
    audit = {
        "source_record": str(record_path), "source_record_sha256": sha(record_path),
        "model_source": str(source), "model_sha256": sha(source),
        "model_matches_archived_sha256": True,
        "mesh_sources": [{"path": str(path), "sha256": sha(path)} for path in mesh_files],
        "mujoco_version": mujoco.__version__, "pyopengl_version": OpenGL.__version__,
        "backend": os.environ["MUJOCO_GL"], "egl_opaque_handle_compatibility_applied": EGL_TYPEDEF_COMPAT,
        "render_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "case_id": record["case_id"], "initial_q": record["initial_q"],
        "obstacles": record["obstacles"], "obstacle_count": len(record["obstacles"]),
        "target_position": record["target_position"], "target_rotation": record["target_rotation"],
        "maximum_frame_position_residual_m": residual,
        "camera": {"azimuth": args.azimuth, "elevation": args.elevation,
                   "distance": args.distance, "lookat": camera.lookat.tolist(), "fovy": 35},
        "appearance_only_changes": ["white background; floor hidden", "studio lights", "transparent spherical obstacles",
                                    "hide source demonstration obstacles and coordinate decorations",
                                    "green goal marker and RGB target orientation triad", "hide collision mesh layer"],
        "physics_steps": 0, "simulation_time": float(data.time), "new_training": False,
        "image": str(output), "image_sha256": sha(output), "image_pixels": [width, height],
        "visible_content_bbox_pixels": visible_bbox,
        "visible_edge_margins_left_top_right_bottom_pixels": edge_margins,
        "framing_visual_check": "Robot, all obstacle sphere extents and target pose retained; camera zoom only, no image crop",
        "labels_in_image": False,
        "camera_azimuth_candidates_reviewed": [-45, 45, 135],
        "reproduce": f"{repo}/.venv/bin/python scripts/render_panda_slide.py --repo-root {repo} --azimuth {args.azimuth} --elevation {args.elevation} --distance {args.distance}",
        "license": "MuJoCo Menagerie Franka Emika Panda mesh assets: Apache-2.0, see evidence/MENAGERIE_LICENSE",
    }
    (ROOT / "evidence/mujoco_render_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ["image", "image_pixels", "maximum_frame_position_residual_m", "camera", "physics_steps"]}, indent=2))


if __name__ == "__main__":
    main()

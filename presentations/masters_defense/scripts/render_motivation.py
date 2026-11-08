"""Render two static, illustrative Panda workcells for motivation slides.

The robot configuration and original meshes are archived research inputs; the
workpiece, worktable and fixtures are explanatory geometry, not an evaluated
scene. This script performs forward kinematics only, never a dynamics rollout.
Run with the repository's .venv Python. Labels belong in native Beamer text.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

from render_panda_slide import (ROOT, build_model, locate_repository, numbers,
                                sha, mujoco, np, Image, EGL_TYPEDEF_COMPAT)

WIDTH, HEIGHT = 1400, 1080
CAMERA = {"azimuth": 110., "elevation": -24., "distance": 2.7,
          "lookat": [.30, .03, .18]}
GREEN = [0., .55, .40, 1.]
ORANGE = [.82, .37, .12, 1.]


def construct(repo, record):
    _, source, meshes, original_xml = build_model(repo, record, WIDTH, HEIGHT)
    tree = ET.fromstring(original_xml)
    world = tree.find('worldbody')
    for geom in list(world.findall('geom')):
        if geom.attrib.get('name', '').startswith('slide_obstacle_'):
            world.remove(geom)
    geometry = []

    def box(name, pos, size, rgba, stage=1):
        attr = {'name': name, 'type': 'box', 'pos': numbers(pos),
                'size': numbers(size), 'rgba': numbers(rgba),
                'contype': '0', 'conaffinity': '0'}
        ET.SubElement(world, 'geom', attr)
        geometry.append(dict(name=name, pos=pos, half_size=size, rgba=rgba,
                             first_visible_stage=stage))

    # One consistent table, task platform and workpiece in both images.
    box('motivation_table', [.28, .02, -.055], [.74, .50, .055], [.87, .89, .91, 1])
    box('motivation_table_trim', [.28, .02, -.12], [.73, .49, .010], [.36, .42, .48, 1])
    box('motivation_task_pedestal', [.73, -.13, .17], [.14, .14, .17], [.36, .48, .58, 1])
    box('motivation_task_top', [.73, -.13, .35], [.16, .16, .018], [.62, .69, .74, 1])
    box('motivation_workpiece', [.73, -.13, .408], [.070, .055, .040], GREEN)

    # Additional surrounding fixtures. Their visibility changes, not the robot.
    box('motivation_fixture_rear', [.28, .37, .31], [.38, .065, .31], [.74, .79, .82, 1], 2)
    box('motivation_fixture_rear_cap', [.28, .37, .638], [.40, .080, .018], ORANGE, 2)
    box('motivation_fixture_column', [-.26, -.05, .28], [.065, .11, .28], [.74, .79, .82, 1], 2)
    box('motivation_fixture_column_cap', [-.26, -.05, .575], [.080, .12, .015], ORANGE, 2)
    box('motivation_fixture_right', [.90, .24, .22], [.10, .085, .22], [.74, .79, .82, 1], 2)
    box('motivation_fixture_right_cap', [.90, .24, .455], [.11, .095, .015], ORANGE, 2)
    # The same input meshes and forward kinematics are retained verbatim.
    xml = ET.tostring(tree, encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml), source, meshes, xml, geometry


def world_marker(scene, pos):
    # A green goal ring above the workpiece, with no implied controller path.
    radius=.085
    for index in range(48):
        theta0=2*np.pi*index/48
        theta1=2*np.pi*(index+1)/48
        start=pos+np.array([radius*np.cos(theta0), radius*np.sin(theta0), 0.])
        end=pos+np.array([radius*np.cos(theta1), radius*np.sin(theta1), 0.])
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                           np.asarray(GREEN, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .004, start, end)
        scene.ngeom += 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root')
    parser.add_argument('--azimuth', type=float, default=CAMERA['azimuth'])
    parser.add_argument('--preview-dir', type=Path)
    args=parser.parse_args()
    repo=locate_repository(args.repo_root)
    record_path=ROOT/'evidence/panda_example.json'
    record=json.loads(record_path.read_text())
    model, source, meshes, xml, geometry=construct(repo, record)
    data=mujoco.MjData(model)
    data.qpos[:]=record['initial_q']
    mujoco.mj_forward(model, data)
    positions=[]
    for name in record['frame_names']:
        try: positions.append(data.body(name).xpos.copy())
        except KeyError: positions.append(data.site(name).xpos.copy())
    residual=float(np.max(np.abs(np.array(positions)-record['initial_frame_positions'])))
    np.testing.assert_allclose(positions, record['initial_frame_positions'], atol=1e-12)
    assert sha(source)==record['model_sha256']
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=CAMERA['lookat']
    camera.azimuth=args.azimuth
    camera.elevation=CAMERA['elevation']
    camera.distance=CAMERA['distance']
    option=mujoco.MjvOption()
    option.geomgroup[3]=0
    option.sitegroup[:]=0
    original_rgba=model.geom_rgba.copy()
    outputs=[]
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        def render(stage, path, azimuth):
            camera.azimuth=azimuth
            model.geom_rgba[:]=original_rgba
            for geom in geometry:
                if geom['first_visible_stage']>stage:
                    model.geom(geom['name']).rgba[3]=0
            renderer.update_scene(data, camera=camera, scene_option=option)
            world_marker(renderer.scene, np.array([.73, -.13, .49]))
            pixels=renderer.render()
            Image.fromarray(pixels).save(path)
            nonwhite=np.min(pixels, axis=2)<245
            yy,xx=np.nonzero(nonwhite)
            return [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
        if args.preview_dir:
            args.preview_dir.mkdir(exist_ok=True, parents=True)
            for az in [40,70,110,140]:
                render(2, args.preview_dir/f'motivation_az{az}.png', az)
        for stage, suffix in [(1,'task'),(2,'safety')]:
            path=ROOT/f'Fig/motivation_{suffix}.png'
            bbox=render(stage,path,args.azimuth)
            assert min(bbox[0], bbox[1], WIDTH-1-bbox[2], HEIGHT-1-bbox[3]) >= 50, 'Concept workcell clipped by the viewport'
            outputs.append(dict(stage=stage,path=str(path),sha256=sha(path),
                                visible_content_bbox_pixels=bbox))
    assert data.time==0 and np.all(data.qvel==0)
    audit={
        'purpose':'Static motivation concept illustration; not an experimental scene or a controller result.',
        'source_record':'panda_example.json',
        'source_sha256':{'panda_example.json':sha(record_path)},
        'source_configuration':record['initial_q'],
        'model_source':str(source),'model_sha256':sha(source),
        'mesh_sources':[{'path':str(p),'sha256':sha(p)} for p in meshes],
        'maximum_frame_position_residual_m':residual,
        'geometry':geometry,
        'goal_ring_center':[.73,-.13,.49],
        'same_robot_configuration_in_both_images':True,
        'same_camera_in_both_images':True,
        'camera':dict(CAMERA,azimuth=args.azimuth),
        'image_pixels':[WIDTH,HEIGHT],
        'outputs':outputs,'physics_steps':0,'simulation_time':float(data.time),
        'not_claimed':['successful reach','collision-free motion','evaluated workcell','planned path','safety certification'],
        'mujoco_version':mujoco.__version__,
        'egl_opaque_handle_compatibility_applied':EGL_TYPEDEF_COMPAT,
        'render_xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
        'labels_in_image':False,
        'license':'MuJoCo Menagerie Franka Emika Panda meshes: Apache-2.0 (evidence/MENAGERIE_LICENSE); new workcell primitives authored for this deck.',
        'reproduce':f'{repo}/.venv/bin/python scripts/render_motivation.py --repo-root {repo} --azimuth {args.azimuth}',
    }
    (ROOT/'evidence/motivation_visual_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'outputs':outputs,'camera':audit['camera'],'physics_steps':0},indent=2))


if __name__=='__main__': main()

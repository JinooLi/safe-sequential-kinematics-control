"""Render archived initial/final Panda configurations, without a new rollout.

The three panels are independent static views of one saved scene. The second
and third panels show each run's OWN endpoint, not states at a common time.
No intermediate configurations, paths or trajectories are reconstructed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/masters-defense-recovery-mpl')
from render_panda_slide import build_model, add_target, locate_repository, sha
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / 'evidence', ROOT / 'Fig'
FONT_SIZE = 24  # One source size for all visible labels; scaled with the vector figure.
plt.rcParams.update({
    "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "figure.titlesize": FONT_SIZE,
    'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
    'pgf.preamble': r'\usepackage{kotex}\usepackage[english]{babel}\usepackage[T1]{fontenc}\usepackage{lmodern}\renewcommand{\familydefault}{\sfdefault}',
    'font.family': 'sans-serif', 'font.size': FONT_SIZE,
    'text.color': '#243447', 'axes.unicode_minus': False,
})


def orientation_error(target_wxyz, rotation):
    xyzw = Rotation.from_matrix(rotation).as_quat()
    current = xyzw[[3, 0, 1, 2]]
    desired = np.asarray(target_wxyz, dtype=float)
    desired /= np.linalg.norm(desired)
    if desired @ current < 0:
        desired = -desired
    vector = (current[0] * desired[1:] - desired[0] * current[1:]
              - np.cross(desired[1:], current[1:]))
    return float(np.linalg.norm(vector))


def project_target(scene, target, width, height):
    """Project a world point using MuJoCo's mono camera and perspective frustum.

    MuJoCo stores left/right GL cameras; mono rendering uses their midpoint.
    Pixel coordinates use the image's upper-left origin, matching imshow.
    """
    cameras = list(scene.camera)
    position = np.mean([camera.pos for camera in cameras], axis=0)
    forward = np.mean([camera.forward for camera in cameras], axis=0)
    up = np.mean([camera.up for camera in cameras], axis=0)
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    offset = np.asarray(target) - position
    depth = float(offset @ forward)
    assert depth > 0 and all(not camera.orthographic for camera in cameras)
    near = float(np.mean([camera.frustum_near for camera in cameras]))
    top = float(np.mean([camera.frustum_top for camera in cameras]))
    bottom = float(np.mean([camera.frustum_bottom for camera in cameras]))
    center = float(np.mean([camera.frustum_center for camera in cameras]))
    half_width = (top - bottom) * .5 * width / height
    x_near = float(offset @ right) * near / depth
    y_near = float(offset @ up) * near / depth
    pixel = [(x_near - center + half_width) / (2 * half_width) * width,
             (top - y_near) / (top - bottom) * height]
    return pixel, {'position': position.tolist(), 'forward': forward.tolist(),
                   'up': up.tolist(), 'frustum_near': near, 'frustum_top': top,
                   'frustum_bottom': bottom, 'frustum_center': center,
                   'target_depth': depth, 'image_pixels': [width, height]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root')
    parser.add_argument('--azimuth', type=float, default=45)
    parser.add_argument('--elevation', type=float, default=-20)
    parser.add_argument('--distance', type=float, default=3.5)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    repo = locate_repository(args.repo_root)
    record_path = DATA / 'recovery_scene.json'
    record = json.loads(record_path.read_text())
    context = record['context']
    fixed, learned = sorted(record['results'], key=lambda row: row['pair'])
    assert fixed['pair'] == [3, 5] and learned['pair'] == [5, 6]
    assert not fixed['success'] and fixed['reason'] == 'timeout'
    assert fixed['stage_transitions'] == [] and fixed['final_stage'] == 'link3_pose'
    assert learned['success'] and learned['reason'] == 'goal_reached'
    assert abs(fixed['elapsed'] - 120) < 1e-8
    assert abs(learned['elapsed'] - 23.42) < 1e-8
    assert fixed['case_sha256'] == learned['case_sha256']
    assert fixed['ik_sha256'] == learned['ik_sha256']
    with (DATA / 'selector_choices.csv').open() as stream:
        choice = next(row for row in csv.DictReader(stream)
                      if row['case_id'] == record['case_id'] and row['split'] == 'test'
                      and row['family'] == 'baseline' and row['seed'] == '11')
    assert choice == record['selector_choice']
    assert int(choice['selected_pair_index']) == 9 and int(choice['success']) == 1
    geometry = {'obstacles': [item['position'] + [item['radius']]
                              for item in context['obstacles']]}
    width, height = 1000, 1000
    model, source, mesh_files, xml = build_model(repo, geometry, width, height)
    assert sha(source) == json.loads((DATA / 'panda_example.json').read_text())['model_sha256']
    target = np.asarray(context['target_position'])
    quaternion = np.asarray(context['target_quaternion'])
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.05, -.03, .5]
    camera.distance, camera.elevation, camera.azimuth = args.distance, args.elevation, args.azimuth
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    states = [context['initial_q'], fixed['final_q'], learned['final_q']]
    data = mujoco.MjData(model)
    errors, pixels, bounds, projections, camera_records = [], [], [], [], []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index, q in enumerate(states):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            actual_position = data.site('gripper_tip').xpos.copy()
            actual_rotation = data.site('gripper_tip').xmat.reshape(3, 3).copy()
            error = {'position_m': float(np.linalg.norm(actual_position - target)),
                     'orientation_quaternion_vector': orientation_error(quaternion, actual_rotation),
                     'end_effector_position': actual_position.tolist()}
            if index:
                saved = [fixed, learned][index - 1]
                np.testing.assert_allclose(error['position_m'], saved['final_position_error'], atol=1e-12)
                np.testing.assert_allclose(error['orientation_quaternion_vector'], saved['final_orientation_error'], atol=1e-12)
            errors.append(error)
            renderer.update_scene(data, camera=camera, scene_option=option)
            add_target(renderer.scene, target, rotation)
            pixel, camera_record = project_target(renderer.scene, target, width, height)
            projections.append(pixel)
            camera_records.append(camera_record)
            rgb = renderer.render().copy()
            pixels.append(rgb)
            ys, xs = np.nonzero(rgb.min(axis=2) < 248)
            bounds.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
            assert data.time == 0 and np.all(data.qvel == 0)
            if args.preview_dir:
                args.preview_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(args.preview_dir / f'state_{index}.png')
    for bound in bounds:
        assert min(bound[0], bound[1], width - 1 - bound[2], height - 1 - bound[3]) > 15
    np.testing.assert_allclose(projections, np.tile(projections[0], (3, 1)), atol=1e-10)
    marker_checks = []
    for rgb, pixel in zip(pixels, projections):
        x, y = np.rint(pixel).astype(int)
        roi = rgb[y-8:y+9, x-8:x+9].astype(float)
        green = (roi[:, :, 1] > roi[:, :, 0] + 20) & (roi[:, :, 1] > roi[:, :, 2] + 10)
        marker_checks.append(int(green.sum()))
        assert marker_checks[-1] > 0, 'Projected target does not overlap the rendered green goal'
    titles = ['Initial configuration', 'Example pair (3,5)', 'Learned selection (5,6)']
    captions = ['$t=0$ s', '$t=120$ s: goal not reached', '$t=23.42$ s: goal reached']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, rgb, title, caption, color, pixel in zip(axes, pixels, titles, captions,
                                             ['#243447', '#FF0000', '#008000'], projections):
        ax.imshow(rgb)
        ax.axis('off')
        # Native vector annotation, anchored to the projected world goal.
        ax.add_patch(Circle(pixel, radius=39, fill=False, edgecolor='#00845b', linewidth=2.2))
        ax.annotate('EE goal', xy=(pixel[0]-22, pixel[1]-32),
                    xytext=(550, 115), ha='center', va='center', fontsize=FONT_SIZE,
                    color='#00845b', arrowprops={'arrowstyle': '-', 'color': '#00845b',
                                               'linewidth': 1.8, 'shrinkA': 5, 'shrinkB': 0})
        ax.set_title(title, fontsize=FONT_SIZE, pad=9, color=color)
        ax.text(.5, -.035, caption, transform=ax.transAxes,
                ha='center', va='top', fontsize=FONT_SIZE, color=color)
    fig.subplots_adjust(left=.005, right=.995, bottom=.12, top=.85, wspace=.035)
    output = FIG / 'recovery_comparison.pdf'
    fig.savefig(output, bbox_inches='tight', pad_inches=.035)
    plt.close(fig)
    subprocess.run(['pdftoppm', '-r', '160', '-singlefile', '-png', str(output),
                    str(FIG / 'recovery_comparison')], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    audit = {
        'source_sha256': {name: sha(DATA / name) for name in
                          ['recovery_scene.json', 'selector_choices.csv', 'panda_example.json']},
        'case_id': record['case_id'], 'split': 'test', 'model_family': 'baseline', 'model_seed': 11,
        'source_records': record['source_records'],
        'model_source': str(source), 'model_sha256': sha(source),
        'model_matches_archived_sha256': True,
        'mesh_sources': [{'path': str(path), 'sha256': sha(path)} for path in mesh_files],
        'helper_source_sha256': sha(ROOT / 'scripts/render_panda_slide.py'),
        'render_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
        'font_rendering': 'pdfLaTeX + ko.TeX + Latin Modern; PNG rendered from PDF',
        'renderer': 'MuJoCo static forward kinematics with original Panda meshes',
        'mujoco_version': mujoco.__version__,
        'common_camera': {'azimuth': args.azimuth, 'elevation': args.elevation,
                          'distance': args.distance, 'lookat': camera.lookat.tolist(), 'fovy': 35},
        'same_initial_configuration_obstacles_goal_and_ik': True,
        'common_obstacles': geometry['obstacles'], 'target_position': target.tolist(),
        'target_rotation': rotation.tolist(), 'configurations': states,
        'panel_times_seconds': [0, fixed['elapsed'], learned['elapsed']],
        'final_panels_are_own_endpoints_not_common_time': True,
        'intermediate_configurations_or_trajectories_invented': False,
        'fixed_result': {k: fixed[k] for k in ['pair', 'success', 'reason', 'elapsed', 'final_stage', 'stage_transitions', 'final_position_error', 'final_orientation_error']},
        'learned_result': {k: learned[k] for k in ['pair', 'success', 'reason', 'elapsed', 'final_stage', 'stage_transitions', 'final_position_error', 'final_orientation_error']},
        'rendered_end_effector_errors': errors,
        'final_errors_match_archive_atol': 1e-12,
        'physics_steps': 0, 'new_rollout': False, 'new_training': False,
        'visible_content_bbox_pixels': bounds,
        'goal_annotation': {'native_vector': True, 'label': 'EE goal',
                            'projected_pixel_centers': projections, 'circle_radius_pixels': 39,
                            'projection': 'Perspective projection from the midpoint MuJoCo GL camera',
                            'camera_records': camera_records,
                            'green_goal_pixels_within_8px_of_projection': marker_checks,
                            'same_goal_projection_verified': True},
        'interpretation': 'Selected recovery example, showing differing terminal configurations. Not a causal proof or a representative statistical sample.',
        'figure_files': ['recovery_comparison.pdf', 'recovery_comparison.png'],
        'figure_sha256': {name: sha(FIG / name) for name in ['recovery_comparison.pdf', 'recovery_comparison.png']},
        'script_sha256': sha(Path(__file__)),
    }
    (DATA / 'recovery_comparison_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: audit[key] for key in ['case_id', 'panel_times_seconds', 'rendered_end_effector_errors', 'physics_steps', 'visible_content_bbox_pixels']}, indent=2))


if __name__ == '__main__':
    main()

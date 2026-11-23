"""Check built slides, copied evidence, page geometry and expected figures."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from speaker_notes import read_speaker_notes

ROOT = Path(__file__).resolve().parents[1]
report = {"evidence_hashes_checked": 0, "pdfs": {}, "issues": []}
plan = json.loads((ROOT / "evidence/story_plan.json").read_text())
page_by_label = {s['label']: s['page'] for s in plan['slides']}
main_pages = plan["main_slides"] + plan["appendix_slides"]
presenter_pages = main_pages + plan["main_slides"]
for item in json.loads((ROOT / "evidence/sources.json").read_text()):
    p = ROOT / item["copy"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"], p
    report["evidence_hashes_checked"] += 1

for stem, expected in [("main", main_pages), ("presenter", presenter_pages)]:
    pdf = ROOT / "build" / f"{stem}.pdf"
    log = (ROOT / "build" / f"{stem}.log").read_text(errors="replace")
    bad = [line for line in log.splitlines()
           if any(x in line for x in ["Overfull", "Missing character", "LaTeX Error", "undefined", "Font Warning"])]
    assert not bad, bad
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)", info)[1])
    assert pages == expected, (stem, pages)
    w,h = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
    assert abs(w/h - 4/3) < .001
    xml = subprocess.check_output(["pdftotext", "-bbox", str(pdf), "-"], text=True)
    # Some TeX extensible math delimiters extract as XML-forbidden C0 glyphs.
    # Remove only those control codes; keep their bounding boxes for the check.
    xml = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", xml)
    root = ET.fromstring(xml)
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    clipped = []
    words_total = 0
    for page_no,page in enumerate(root.findall(".//x:page",ns),1):
        width,height=float(page.attrib['width']),float(page.attrib['height'])
        words=page.findall(".//x:word",ns)
        assert words, (stem,page_no,"blank page")
        words_total += len(words)
        for word in words:
            x0,y0,x1,y1=[float(word.attrib[k]) for k in ['xMin','yMin','xMax','yMax']]
            if x0 < -.5 or y0 < -.5 or x1 > width+.5 or y1 > height+.5:
                clipped.append([page_no,word.text,[x0,y0,x1,y1]])
    assert not clipped, (stem,clipped)
    text = subprocess.check_output(["pdftotext",str(pdf),"-"],text=True)
    for value in ["45.50", "55.20", "58.70", "60.90", "21,423"]:
        assert value in text, (stem,value)
    page_texts = text.split("\f")[:-1]
    assert len(page_texts) == pages
    note_page_indexes = set(range(1, 2 * plan['main_slides'], 2)) if stem == 'presenter' else set()
    for index, page_text in enumerate(page_texts):
        has_korean = bool(re.search(r"[가-힣]", page_text))
        assert has_korean == (index in note_page_indexes), (stem, index + 1, "Unexpected page language")
    report["pdfs"][stem] = {"pages":pages,"page_size_points":[w,h],
        "extracted_words":words_total,"out_of_page_text_boxes":len(clipped),
        "layout_or_font_errors":bad,"sha256":hashlib.sha256(pdf.read_bytes()).hexdigest()}

source=(ROOT/'main.tex').read_text()
# main.tex contains slide copy only; the rehearsal script is separate.
assert r'\talknote' not in source and r'\note{' not in source
visible_source=source
assert r'\src{' not in source
assert not re.search(r'(했다|않았다|한다|된다|남는다|있다)\.', visible_source)
font_report=subprocess.check_output(['pdffonts',str(ROOT/'build/main.pdf')],text=True)
assert 'DejaVu' not in font_report and 'CID TrueType' not in font_report
assert 'LMSans' in font_report
report['style_checks']={'visible_small_print_removed':True,'english_screen_copy':True,
    'latex_fonts_in_graphs':True, 'user_block_title_colors_preserved':True,
    'clf_shapes_page':page_by_label['planar-clf'],'clf_time_page':page_by_label['planar-why'],
    'ik_pose_overlay_page':page_by_label['planar-setup'],
    'planar_setup_pages':[page_by_label[k] for k in ['planar-scene','planar-setup']]}
notes=read_speaker_notes(ROOT/'speaker-notes.tex')
labels=re.findall(r'\\begin\{frame\}\[[^]]*\blabel=([^,\]]+)', source)
assert len(labels)==len(set(labels))==len(notes)==plan["main_slides"]
assert set(labels)==set(notes)
seconds=sum(item[0] for item in notes.values())
assert seconds==plan["speech_seconds"]==1140
assert labels == [s["label"] for s in plan["slides"]]
assert all(notes[s["label"]][0] == s["seconds"] for s in plan["slides"])
assert all(re.search(r"[가-힣]", text) for _, text in notes.values())
report['language_checks'] = {'audience_slides': 'English', 'speaker_notes': 'Korean',
    'presenter_slide_pages': 'English', 'presenter_note_pages': 'Korean',
    'korean_speaker_notes': len(notes)}
background = json.loads((ROOT/'evidence/background_progress_audit.json').read_text())
assert labels[3:3 + len(background['background_order'])] == background['background_order']
main_source = source.split(r'\begin{document}',1)[1].split(r'\appendix',1)[0]
main_blocks = re.findall(r'\\begin\{frame\}(.*?)\\end\{frame\}', main_source, re.S)
takeaway_labels = []
for block in main_blocks:
    if r'\takeaway{' not in block:
        continue
    takeaway_labels.append(re.search(r'\blabel=([^,\]]+)', block)[1])
    body = block.split(r'\begin{slidebody}', 1)[1].lstrip()
    body = re.sub(r'^(?:\\vspace\*?\{[^}]*\}\s*)+', '', body)
    assert body.startswith(r'\takeaway{'), takeaway_labels[-1]
assert takeaway_labels == background['retained_main_takeaways']
assert r'{block body}\color{black}\textbf{#1}' in source
assert r'\input{progress-style.tex}' in source
assert re.findall(r'\\defensepart\{([^}]+)\}', source) == [p['name'] for p in plan['progress_parts']]
for part in plan['progress_parts']:
    for page in range(part['first_page'],part['last_page']+1):
        assert plan['slides'][page-1]['progress_part'] == part['name']
report['background_and_navigation'] = {'background_order':background['background_order'],
    'main_takeaways':len(takeaway_labels),'takeaways_at_top_in_black':True,
    'progress_parts':[p['name'] for p in plan['progress_parts']],
    'cover_excluded_from_progress':True,'appendix_separate':True,
    'indicator':'one dot per content slide', 'dots':plan['main_slides']-1}
progress_data=(ROOT/'progress-data.tex').read_text()
assert re.findall(r'\\defense@group\{([^}]+)\}\{(\d+)\}\{(\d+)\}',progress_data) == [(p['name'],str(p['first_page']),str(p['last_page'])) for p in plan['progress_parts']]
assert sum(p['last_page']-p['first_page']+1 for p in plan['progress_parts'])==plan['main_slides']-1
# Numbered subtitles retain the thesis heading and terminology mapping.
number_audit = json.loads((ROOT/'evidence/thesis_section_numbers.json').read_text())
assert number_audit['generated_files_checked']['all_heading_names_levels_numbers_and_order_match_current_source']
for item in number_audit['sources']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
for item in json.loads((ROOT/'evidence/chapter_mapping.json').read_text())['mapping']:
    number = number_audit['heading_to_number'].get(item['thesis_heading'])
    assert item['thesis_number'] == number
    assert item['display_subtitle'] == ((number + ' ') if number else '') + item['subtitle']
continuations = plan.get("bibliography_continuation_pages", 0)
assert len(re.findall(r"\\begin\{frame\}",source)) + continuations == main_pages
for rel, expected_sha in json.loads((ROOT/"evidence/story_figure_audit.json").read_text())["source_sha256"].items():
    assert hashlib.sha256((ROOT/"evidence"/rel).read_bytes()).hexdigest() == expected_sha, rel
for rel, expected_sha in json.loads((ROOT/"evidence/concept_figure_audit.json").read_text())["source_sha256"].items():
    assert hashlib.sha256((ROOT/"evidence"/rel).read_bytes()).hexdigest() == expected_sha, rel

setup_audit = json.loads((ROOT/"evidence/planar_setup_audit.json").read_text())
for rel, expected_sha in setup_audit["source_sha256"].items():
    assert hashlib.sha256((ROOT/"evidence"/rel).read_bytes()).hexdigest() == expected_sha, rel
progressive = json.loads((ROOT/'evidence/progressive_planar_audit.json').read_text())
for rel, expected_sha in progressive['source_sha256'].items():
    assert hashlib.sha256((ROOT/'evidence'/rel).read_bytes()).hexdigest() == expected_sha, rel
assert progressive['identical_geometry_transform'] and progressive['all_geometry_inside_axes']
assert progressive['first_three_states_are_identical']
assert progressive['max_selected_ik_target_fk_residual_m'] < 1e-12
assert not progressive['new_simulation_or_training']
for stage in progressive['stages']:
    for rel in stage['figure_files']:
        assert (ROOT/rel).is_file()
        if rel.endswith('.pdf') and not Path(rel).name.startswith('progressive_planar_01_'):
            assert '{'+Path(rel).name+'}' in source
motivation = json.loads((ROOT/'evidence/motivation_visual_audit.json').read_text())
assert motivation['same_camera_in_both_images'] and motivation['same_robot_configuration_in_both_images']
assert motivation['physics_steps'] == 0 and motivation['maximum_frame_position_residual_m'] < 1e-10
for item in motivation['outputs']:
    name = Path(item['path']).name
    assert hashlib.sha256((ROOT/'Fig'/name).read_bytes()).hexdigest() == item['sha256']
    # Keep hashes of the archived illustration pair; only safety remains on screen.
assert '{motivation_safety.png}' in source
photos = json.loads((ROOT/'evidence/motivation_photo_audit.json').read_text())
for item in photos['assets']:
    assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest() == item['sha256']
    assert '{'+Path(item['file']).name+'}' in source
    assert item['original_bytes_preserved']
report['visual_story_checks'] = {'progressive_planar_stages':len(progressive['stages']),
    'fixed_geometry_transform':True,'archived_case':progressive['case_id'],
    'motivation_real_photographs':len(photos['assets']),
    'motivation_photograph_source':photos['source_page'],
    'motivation_safety_static_illustration':True}

# Recovery example uses archived endpoints, not an invented intermediate path.
recovery = json.loads((ROOT/"evidence/recovery_comparison_audit.json").read_text())
for rel, expected_sha in recovery["source_sha256"].items():
    assert hashlib.sha256((ROOT/"evidence"/rel).read_bytes()).hexdigest() == expected_sha, rel
for rel, expected_sha in recovery["figure_sha256"].items():
    assert hashlib.sha256((ROOT/"Fig"/rel).read_bytes()).hexdigest() == expected_sha, rel
assert recovery["physics_steps"] == 0 and not recovery["new_rollout"]
assert recovery["same_initial_configuration_obstacles_goal_and_ik"]
assert recovery["final_panels_are_own_endpoints_not_common_time"]
assert not recovery["intermediate_configurations_or_trajectories_invented"]
assert not recovery["fixed_result"]["success"] and recovery["learned_result"]["success"]
for index, result_key in [(1, "fixed_result"), (2, "learned_result")]:
    rendered = recovery["rendered_end_effector_errors"][index]
    archived = recovery[result_key]
    assert abs(rendered["position_m"] - archived["final_position_error"]) < 1e-12
    assert abs(rendered["orientation_quaternion_vector"] - archived["final_orientation_error"]) < 1e-12
assert "{recovery_comparison.pdf}" in source
learning_start = labels.index('selector-training')
assert labels[learning_start:learning_start+3] == ['selector-training','selector-features','selector-model']
claims = json.loads((ROOT/"evidence/refinement_claims_audit.json").read_text())
assert hashlib.sha256((ROOT/"질의응답_메모.txt").read_bytes()).hexdigest() == claims["qanda_after_sha256"]
report["refinement_checks"] = {"learning_task_before_architecture": True,
    "normal_transition_explained": "Subtask transition" in source,
    "ik_episode_population_visible": all(x in source for x in ["300 IK episodes", "1,000 IK episodes"]),
    "recovery_case": recovery["case_id"], "recovery_static_endpoint_render": True,
    "recovery_panel_seconds": recovery["panel_times_seconds"]}

# The two-link reduction is introduced through all candidates before a
# training-selected comparison baseline. Keep success and time populations distinct.
two_link = json.loads((ROOT/'evidence/two_link_average_audit.json').read_text())
for item in two_link['input_sources']:
    assert hashlib.sha256((ROOT/item['snapshot']).read_bytes()).hexdigest() == item['sha256']
script_info = two_link['analysis_script']
assert hashlib.sha256((ROOT/script_info['path']).read_bytes()).hexdigest() == script_info['sha256']
assert two_link['verification']['validation_outcomes_checked_against_raw_npz_and_csv'] == 10000
assert two_link['verification']['existing_fixed_3_5_counts_and_paired_means_reproduced']
assert two_link['no_new_rollouts'] and two_link['no_new_training']
two_frame = next(block for block in main_blocks if 'label=two-links]' in block)
success = two_link['success']
paired = two_link['pooled_common_success']
assert all(f'{value:.2f}' in two_frame for value in [
    success['all_links_success_rate_percent'], success['two_link_success_rate_percent'],
    success['two_link_minus_all_links_percentage_points'], paired['all_links_mean_seconds'],
    paired['two_link_mean_seconds'], -paired['relative_reduction_percent']])
assert f"{paired['both_success_records']:,}" in two_frame
assert 'episode--pair outcomes' in two_frame
assert two_link['pooled_time_interpretation']['not_independent_episodes']
assert 'Preset' not in two_frame and 'training' not in two_frame.lower()
test_success_frame = next(block for block in main_blocks if 'label=test-success]' in block)
assert '10-pair mean' in test_success_frame and '{test_success_pair_mean.pdf}' in test_success_frame
assert 'Preset pair' not in source
average_plot = json.loads((ROOT/'evidence/two_link_average_figure_audit.json').read_text())
for rel, sha in average_plot['source_sha256'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == sha
for rel, sha in average_plot['output_sha256'].items():
    assert hashlib.sha256((ROOT/'Fig'/rel).read_bytes()).hexdigest() == sha
assert abs(average_plot['candidate_average']['success_rate_percent'] - success['two_link_success_rate_percent']) < 1e-10
assert '{story_oracle_average.pdf}' in source
progress_source = (ROOT/'progress-style.tex').read_text()
assert progress_source.count(r'\fontsize{7}{8.4}\selectfont\color{black!75}\insertframenumber') == 2
report['two_link_average'] = {
    'success_percent': success['two_link_success_rate_percent'],
    'common_success_episode_pair_records': paired['both_success_records'],
    'unique_episodes_in_time_comparison': paired['unique_common_success_episodes'],
    'paired_means_seconds': [paired['all_links_mean_seconds'], paired['two_link_mean_seconds']],
    'time_reduction_percent': paired['relative_reduction_percent'],
    'main_results_reference': '10-pair mean',
    'page_number_font_pt': 7,
    'raw_outcomes_unchanged': True,
}

# Results use the same candidate-average baseline, with uncertainty clustered
# by episode and completion times conditioned on common success.
mean_results = json.loads((ROOT/'evidence/pair_mean_results_audit.json').read_text())
for item in mean_results['input_sources']:
    assert hashlib.sha256((ROOT/item['snapshot']).read_bytes()).hexdigest() == item['sha256']
analysis = mean_results['analysis_script']
assert hashlib.sha256((ROOT/analysis['path']).read_bytes()).hexdigest() == analysis['sha256']
assert all(mean_results['verification'].values())
assert mean_results['no_new_rollouts'] and mean_results['no_new_training']
assert mean_results['bootstrap']['episodes_per_draw'] == 1000
assert mean_results['bootstrap']['draws'] == 10000
assert mean_results['bootstrap']['resampling_unit'].startswith('episode')
gain = mean_results['success']['learned_vs_pair_mean']
assert all(f'{value:.2f}' in test_success_frame for value in
           [gain['difference_pp'], *gain['bootstrap_95_ci_pp']])
mean_figure = json.loads((ROOT/'evidence/pair_mean_results_figure_audit.json').read_text())
for rel, sha in mean_figure['source_sha256'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == sha
for rel, sha in mean_figure['output_sha256'].items():
    assert hashlib.sha256((ROOT/'Fig'/rel).read_bytes()).hexdigest() == sha
mean_time = mean_results['paired_time']['learned_vs_pair_mean']
appendix_time = source.split(r'\hypertarget{appendix-a6}',1)[1].split(r'\end{frame}',1)[0]
assert all(f'{value:.2f}' in appendix_time for value in
           [mean_time['reference_common_mean_seconds'],mean_time['candidate_common_mean_seconds']])
assert f"{mean_time['both_success_records']:,} episode--pairs" in appendix_time
assert '{selection_gain_pair_mean.pdf}' in source
assert 'Train-best fixed pair' in source and '[-0.30,4.70]' in source
report['pair_mean_results'] = {
    'reference': '10-pair mean', 'success_gain_pp':gain['difference_pp'],
    'bootstrap_95_ci_pp':gain['bootstrap_95_ci_pp'],
    'bootstrap_unit':'episode',
    'paired_time_seconds':[mean_time['reference_common_mean_seconds'],
                           mean_time['candidate_common_mean_seconds']],
    'common_success_episode_pair_records':mean_time['both_success_records'],
    'unique_common_success_episodes':mean_time['unique_common_success_episodes'],
    'train_best_fixed_reference_retained_in_appendix':True,
    'raw_outcomes_unchanged':True,
}

# Subtitles use actual thesis sections/subsections, with explicit title aliases.
chapter_map = json.loads((ROOT/"evidence/chapter_mapping.json").read_text())
subtitles = re.findall(r"\\framesubtitle\{([^{}]*)\}", source)
assert subtitles == [item["display_subtitle"] for item in chapter_map["mapping"]]
assert chapter_map['excluded_frames'] == ['title','questions','appendix-contents']
assert len(subtitles) == main_pages - continuations - len(chapter_map['excluded_frames'])
for item in chapter_map['mapping']:
    if item['heading_level'] == 'bibliography':
        assert item['subtitle'] == 'Bibliography'
        continue
    heading = item['thesis_heading']
    thesis_source = Path(item['source']).read_text()
    pattern = (r'\\' + item['heading_level'] +
               r'(?:\[[^]]*\])?\{' + re.escape(heading) + r'\}')
    assert re.search(pattern, thesis_source), (item['slide'], heading)
    expected = chapter_map['terminology_aliases'].get(item['subtitle'], item['subtitle'])
    assert expected == heading, item
    if item['heading_level'] == 'chapter':
        chapter_frames = {
            'thesis-outline': 'A Sequential Approach to Collision-Free Kinematic Control Based on Control Barrier Functions and Supervised Learning',
            'conclusion': 'Conclusion',
        }
        assert chapter_frames.get(item['slide']) == heading

# The PDF bibliography is generated by BibTeX and every listed source is cited.
citations = {key for group in re.findall(r"\\cite\{([^}]*)\}", source)
             for key in group.split(",")}
assert r"\bibliographystyle{IEEEtran}" in source
assert r"\bibliography{references}" in source
for stem in ["main", "presenter"]:
    bbl = (ROOT/"build"/f"{stem}.bbl").read_text()
    assert citations == set(re.findall(r"\\bibitem\{([^}]*)\}", bbl))
    blg = (ROOT/"build"/f"{stem}.blg").read_text()
    assert "Warning--" not in blg and "error message" not in blg
report["bibliography"] = {"backend":"BibTeX", "style":"IEEEtran",
                          "cited_entries":len(citations),
                          "continuation_pages":continuations}
literature_frame = source.split(r'\begin{frame}[label=literature]', 1)[1].split(r'\end{frame}', 1)[0]
assert all(key in literature_frame for key in [
    'reis2021equilibria', 'tan2024equilibria', 'goncalves2023circulation'])
assert 'ames2017cbf' not in literature_frame and 'li2024rdf' not in literature_frame
rdf_frame = source.split(r'\begin{frame}[label=panda-rdf]', 1)[1].split(r'\end{frame}', 1)[0]
assert 'RDF' in rdf_frame and '{rdf_complex_query_field.pdf}' in rdf_frame
assert r'\tilde d_b(\xi_{ab,k})' in rdf_frame
assert r'\nabla_qh_{ab}^{\top}u\geq-\gamma h_{ab}' in rdf_frame
assert "account for both links' motion" in rdf_frame
assert 'li2024rdf' in rdf_frame.split(r'\framesubtitle', 1)[0]
# Retain provenance checks for the previous original-paper figure, now archived.
rdf_reference = json.loads((ROOT/'evidence/rdf_original_figure_audit.json').read_text())
assert hashlib.sha256((ROOT/rdf_reference['display_file']).read_bytes()).hexdigest() == rdf_reference['display_sha256']
assert rdf_reference['display_sha256'] == rdf_reference['source_local_sha256']
assert rdf_reference['source_copied_without_modification']
assert not rdf_reference['actual_thesis_controller_result']
assert 'li2024rdf' in rdf_frame
# Preserve reproducibility checks for the earlier, no-longer-displayed schematic.
rdf_figure = json.loads((ROOT/'evidence/rdf_query_figure_audit.json').read_text())
for name, sha in rdf_figure['output_sha256'].items():
    assert hashlib.sha256((ROOT/'Fig'/name).read_bytes()).hexdigest() == sha
assert rdf_figure['surface_query_residual_max'] < 1e-12
assert not rdf_figure['actual_panda_rdf'] and rdf_figure['no_new_simulation']
# The current figure illustrates query evaluation for two non-capsule shapes.
rdf_complex = json.loads((ROOT/'evidence/rdf_complex_query_figure_audit.json').read_text())
for name, sha in rdf_complex['output_sha256'].items():
    assert hashlib.sha256((ROOT/'Fig'/name).read_bytes()).hexdigest() == sha
assert rdf_complex['surface_query_residual_max'] < 1e-10
assert rdf_complex['queries_all_on_source_surface']
assert rdf_complex['queries_all_outside_target']
assert rdf_complex['no_query_motion_arrow']
assert not rdf_complex['actual_panda_rdf'] and rdf_complex['no_new_simulation']
report['literature_and_rdf'] = {'follow_up_control_studies': 3,
    'geometry_reference_used_at_rdf_method': True,
    'query_to_barrier_to_qp_explicit': True, 'field_figure_is_schematic': True,
    'displayed_figure': 'rdf_complex_query_field.pdf',
    'figure_source': 'scripts/build_rdf_complex_query_figure.py',
    'surface_query_residual': rdf_complex['surface_query_residual_max'],
    'queries_on_source_surface': True, 'no_query_motion_arrow': True,
    'archived_original_paper_asset_unchanged': True,
    'archived_schematic_surface_query_residual': rdf_figure['surface_query_residual_max']}
report["section_subtitles"] = {"frames":len(subtitles),
                               "source":"evidence/chapter_mapping.json",
                               "numbered_subtitles":sum(x["thesis_number"] is not None for x in chapter_map["mapping"]),
                               "numbering_evidence":"evidence/thesis_section_numbers.json"}

# Preserve the user's presentation title, which may differ from the thesis title.
# The first method slides define the names.
terminology = json.loads((ROOT/"evidence/terminology.json").read_text())
title = re.search(r"\\title\{([^\n]*)\}", source)[1].replace(r"\\", " ")
assert title == terminology.get("presentation_title", terminology["official_title"])
for definition in terminology["first_definitions"].values():
    assert definition in source, definition
for obsolete in ["장면", "링크", "학습 선택", "전체 링크", "중간 작업", "현재 작업"]:
    assert obsolete not in source, obsolete
audience_text = subprocess.check_output(['pdftotext', str(ROOT/'build/main.pdf'), '-'], text=True)
for obsolete in ['Fixed (3,5)', 'CLF decrease condition', 'IK-admitted', 'Normal switch', 'clearance barrier']:
    assert obsolete not in audience_text, obsolete
assert not re.search(r'\bPanda\b', audience_text)
assert all(x in audience_text for x in ['10-pair mean', 'Example pair', 'Train-best fixed pair',
                                      'Spatial Manipulator', 'Binary Cross Entropy'])
assert 'Preset pair' not in audience_text
assert 'human-inspiration' in labels and 'algorithm-overview' in labels and 'future-work' in labels
render = json.loads((ROOT/"evidence/mujoco_render_audit.json").read_text())
assert hashlib.sha256((ROOT/"Fig/panda_mujoco.png").read_bytes()).hexdigest() == render["image_sha256"]
assert hashlib.sha256((ROOT/"evidence/panda_example.json").read_bytes()).hexdigest() == render["source_record_sha256"]
assert render["maximum_frame_position_residual_m"] < 1e-10
assert render["physics_steps"] == 0 and render["obstacle_count"] == 10
assert "{panda_mujoco.png}" in source
report["terminology"] = {"presentation_title_preserved":True,
                         "title_matches_thesis":title == terminology["official_title"],
                         "first_use_definitions":True,
                         "canonical_method_labels":True,
                         "scene_and_link_consistency":True}
report["panda_render"] = {"backend":"MuJoCo", "static_archived_configuration":True,
                         "image_pixels":render["image_pixels"],
                         "physics_steps":0,
                         "max_frame_residual_m":render["maximum_frame_position_residual_m"]}
# Synchronize current Method definitions without reinterpreting archived rollouts.
notation = json.loads((ROOT/'reviews/notation_sync_revision.json').read_text())
notation_figures = json.loads((ROOT/'evidence/notation_figure_audit.json').read_text())
# Later thesis edits are accepted only when explicitly rechecked; preserve the original audit.
current_notation_hashes = dict(notation['thesis_source_sha256'])
for review_name in ['capsule_diagram_revision', 'subtask_terminology_revision']:
    recheck = json.loads((ROOT/f'reviews/{review_name}.json').read_text())
    for item in recheck['thesis_source_recheck']:
        rel = item['path']
        assert current_notation_hashes[rel] == item['previous_sha256'], rel
        current_notation_hashes[rel] = item['current_sha256']
for rel, expected_sha in current_notation_hashes.items():
    assert hashlib.sha256((Path(notation['thesis_root'])/rel).read_bytes()).hexdigest() == expected_sha, rel
for rel, expected_sha in notation_figures['source_sha256'].items():
    assert hashlib.sha256((ROOT/'evidence'/rel).read_bytes()).hexdigest() == expected_sha, rel
assert notation_figures['trajectory_hashes_match_original_archive']
assert not notation_figures['controller_equivalence_claimed']
assert notation_figures['archive_to_current_scale'] == 0.05
assert notation_figures['orientation_weight'] == 0.5
for case_key in ['case11', 'case30']:
    assert notation_figures[case_key]['max_normalization_error'] < 1e-12
    error_key = 'max_normalized_active_error' if case_key == 'case11' else 'max_normalized_active_clf_error'
    assert notation_figures[case_key][error_key] < 1e-12
assert notation_figures['case30']['max_normalized_direct_clf_error'] < 1e-12
assert not notation['planar_re_evaluation']['new_rollout']
assert not notation['planar_re_evaluation']['source_trajectories_modified']
assert not notation['planar_re_evaluation']['success_labels_modified']
assert not notation['planar_re_evaluation']['completion_times_modified']
assert all(token in source for token in [
    r'T_{\mathrm d}', r'\delta_{\mathrm p}', r'\delta_{\mathrm o}',
    r'\mathcal S=(1,2,3)', r'\subtask{1}\to\cdots\to\subtask{L+1}',
    r'V_3=V_{\mathrm ee}', r'w_{\mathrm o}', r'\Omega',
    r'N_{\mathrm{tr}}', r'\lambda_{\mathrm{sel}}=3'])
full_qp = source.split(r'\begin{frame}{A12. Common CLF--CBF QP}', 1)[1].split(r'\end{frame}', 1)[0]
assert all(token in full_qp for token in [r'V_i', r'c_i', r'\alpha_j', r'v_{\min,j}', r'v_{\max,j}'])
assert not any(token in full_qp for token in [r'V_s', r'c_V', r'\gamma_J'])
# The thesis input limits remain explicit in all three method stages.
# The background slide introduces only the basic objective, CLF, and CBF.
common_qp = source.split(r'\newcommand{\taskqp}[3]', 1)[1].split(r'\input{progress-style.tex}', 1)[0]
normalized_qp = re.sub(r'\s+', '', common_qp)
assert all(token in normalized_qp for token in [
    r'-\alpha_j(q_j-q_{j,\min})\lequ_j\leq\alpha_j(q_{j,\max}-q_j)',
    r'v_{\min,j}\lequ_j\leqv_{\max,j}', r'j=1,\ldots,n', r'r=1,\ldots,B'])
assert r'\mathcal U' not in common_qp
qp_frames = {
    'planar-sequential': r'\taskqp{1}{EmphasisBlue}{c_1}',
    'planar-link2': r'\taskqp{2}{EmphasisOrange}{c_2}',
    'planar-ee': r'\taskqp{3}{EmphasisGreen}{c_3}',
}
for label, macro_call in qp_frames.items():
    frame = next(block for block in main_blocks if f'label={label}]' in block)
    assert macro_call in frame, label
report['explicit_qp_input_bounds'] = {
    'main_pages': [page_by_label[label] for label in qp_frames],
    'joint_limit_cbf_bounds': True, 'velocity_bounds': True,
    'all_joint_and_collision_indices_explicit': True,
    'shared_macro_across_stages': True,
}
qanda = (ROOT/'질의응답_메모.txt').read_text()
assert all(token in qanda for token in ['1/20', '새 QP', '첫 두 열', '23차원', '총 8개 subtask'])
report['notation_sync'] = {
    'current_method_sources_checked': len(notation['thesis_source_sha256']),
    'unchanged_figure_sources_checked': len(notation_figures['source_sha256']),
    'planar_re_evaluation_scale': notation_figures['archive_to_current_scale'],
    'max_planar_re_evaluation_error': max(
        notation_figures['case11']['max_normalized_active_error'],
        notation_figures['case30']['max_normalized_active_clf_error'],
        notation_figures['case30']['max_normalized_direct_clf_error']),
    'new_controller_rollout': False,
    'controller_equivalence_claimed': False,
    'raw_experiment_outcomes_preserved': True,
    'mixed_thesis_notation_documented': len(notation['thesis_inconsistencies']),
    'review': 'reviews/notation_sync_revision.json'}

report.update({"main_slides":plan["main_slides"],"appendix_slides":plan["appendix_slides"],"speech_seconds":seconds,
               "transition_buffer_seconds":60,"new_simulations_or_training":False,
               "speaker_notes":"speaker-notes.tex","notes_matched_by_frame_label":len(notes),
               "figure_audit":"evidence/figure_audit.json",
               "story_figure_audit":"evidence/story_figure_audit.json",
               "story_order_matches_user_outline":True})
(ROOT/'evidence/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))

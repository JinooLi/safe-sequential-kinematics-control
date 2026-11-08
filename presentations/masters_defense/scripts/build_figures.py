"""Build talk figures from frozen thesis records; no learning or simulation."""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/masters-defense-mpl")
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Polygon
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "evidence", ROOT / "Fig"
BLUE, GREEN, RED, GRAY = "#205493", "#28735e", "#a84439", "#77818b"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--only", action="append", metavar="FIGURE",
                    help="Regenerate only this figure stem; repeat for multiple figures.")
requested_figures = set(parser.parse_args().only or [])
generated_figures = []
FONT_SIZE = 22  # One source size for all visible labels; scaled with the vector figure.
plt.rcParams.update({
    "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "figure.titlesize": FONT_SIZE,
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{kotex}\usepackage[english]{babel}\usepackage[T1]{fontenc}\usepackage{lmodern}\renewcommand{\familydefault}{\sfdefault}",
    "font.family": "sans-serif", "font.size": FONT_SIZE,
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": "#243447", "text.color": "#243447"})

def save(fig, name):
    if requested_figures and name not in requested_figures:
        plt.close(fig)
        return
    generated_figures.append(name)
    # All glyphs are typeset by the same pdfLaTeX/ko.TeX/Latin Modern setup
    # as the Beamer deck. PNG previews are rendered from that exact PDF.
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.08)
    subprocess.run(["pdftoppm", "-r", "150", "-singlefile", "-png",
                    str(FIG / f"{name}.pdf"), str(FIG / name)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    plt.close(fig)

def rows(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))

def read(name):
    return json.loads((DATA / name).read_text())

def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.15)

audit = {"font_rendering": "pdfLaTeX + ko.TeX + Latin Modern; PNG rendered from PDF"}
planar = read("planar_cases.json")
counts = [sum(c[m]["success"] for c in planar) for m in ["direct", "sequential"]]
assert counts == [216, 253] and len(planar) == 300
fig, ax = plt.subplots(figsize=(8.0, 2.6))
values = np.array(counts) / len(planar) * 100
b = ax.barh(["Direct", "Sequential"], values, color=[GRAY, BLUE], height=.48)
ax.bar_label(b, labels=[f"{v:.2f}%  ({n}/300)" for v,n in zip(values,counts)], padding=7)
ax.invert_yaxis(); ax.set_xlim(0, 112); ax.set_xticks([0,25,50,75,100])
ax.set_xlabel("Success rate (%)"); decorate(ax)
fig.tight_layout(); save(fig, "planar_success")
audit["planar_success_counts"] = counts

case = next(c for c in planar if c["case_id"] == 11)
traces = {m: np.load(DATA / f"case_011_{m}.npz", allow_pickle=False)
          for m in ["direct", "sequential"]}
def fk(q):
    a = np.cumsum(q)
    return np.vstack(([0,0], np.cumsum(np.c_[np.cos(a),np.sin(a)],axis=0)))

# Actual recorded case, not a synthetic illustration or a new rollout.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharex=True, sharey=True)
for ax, (method, color, title) in zip(axes, [("direct",GRAY,"Direct: 35 s timeout"),
                                           ("sequential",BLUE,"Sequential: success at 17.176 s")]):
    tr = traces[method]
    stop = len(tr["t"])-1 if method == "direct" else np.searchsorted(tr["t"], case[method]["completion_time"])
    for obs in case["obstacles"]:
        ax.add_patch(Circle(obs["position"], obs["radius"], facecolor=RED, alpha=.75))
    for idx in np.linspace(0,stop,5,dtype=int)[1:-1]:
        p=fk(tr["q"][idx]); ax.plot(*p.T,"o-",color=color,alpha=.13,lw=3,ms=3)
    p=fk(tr["q"][stop]); ax.plot(*p.T,"o-",color=color,lw=4,ms=5)
    ax.plot(*tr["final_xy"][:stop+1:20].T,color=color,lw=1.1,alpha=.65)
    ax.plot(*case["target"],marker="*",color=GREEN,ms=14,zorder=5)
    ax.plot(0,0,"ks",ms=6)
    ax.set(xlim=(-2.3,3.2),ylim=(-1.6,2.1),aspect="equal",xlabel="x (m)",title=title)
    ax.grid(alpha=.15)
axes[0].set_ylabel("y (m)"); fig.tight_layout(); save(fig,"planar_scene")

fig, axes = plt.subplots(2,2,figsize=(9,4.5),sharex=True)
for method,color,label in [("direct",GRAY,"Direct"),("sequential",BLUE,"Sequential")]:
    tr=traces[method]; t=tr["t"]
    ep=np.linalg.norm(tr["final_xy"]-case["target"],axis=1)
    angle=tr["final_yaw"]-case["target_yaw"]
    er=np.abs(np.arctan2(np.sin(angle),np.cos(angle)))
    ys=[ep,er,tr["barrier"].min(axis=1),tr["active_link"]]
    for ax,y in zip(axes.flat,ys):
        ax.plot(t[::15],y[::15],label=label,color=color,lw=1.6)
for ax, title, tol in zip(list(axes.flat)[:3],["EE position error (m)","EE yaw error (rad)","Minimum recorded barrier (m)"],[.1,.1,-.01]):
    ax.set_title(title,fontsize=FONT_SIZE); ax.axhline(tol,color=RED,ls="--",lw=1)
axes[0,0].legend(frameon=False,fontsize=FONT_SIZE)
axes[1,1].set_title("Active link / final EE task",fontsize=FONT_SIZE)
axes[1,1].set_yticks([1,2,3]); axes[1,1].set_ylim(.7,3.3)
for ax in axes.flat:
    ax.set_xlim(0,35); ax.grid(alpha=.15)
    ax.axvline(17.176,color=GREEN,ls=":",lw=1)
for ax in axes[-1]: ax.set_xlabel("Simulation time (s)")
fig.tight_layout(); save(fig,"planar_trace")

all_rows = rows("all_links_comparison.csv")
test_rows = rows("test_baselines_comparison.csv")
pair_rows = rows("pair_outcomes.csv")
choices = rows("selector_choices.csv")
summary = read("recomputed_summary.json")
rates = {}
for split, baseline in [("validation",all_rows),("test",test_rows)]:
    assert len(baseline)==len({r["case_id"] for r in baseline})==1000
    pr=[r for r in pair_rows if r["split"]==split]
    cr=[r for r in choices if r["split"]==split and r["family"]=="baseline" and r["seed"]=="11"]
    assert len(pr)==10000 and len(cr)==1000
    assert {r['case_id'] for r in baseline} == {r['case_id'] for r in cr} == {r['case_id'] for r in pr}
    pair_index={(r['case_id'],int(r['pair_index'])):r for r in pr}
    for r in cr:
        src=pair_index[(r['case_id'],int(r['selected_pair_index']))]
        assert int(r['success'])==int(src['success'])
    ds=sum(r["direct_120_success"]=="True" for r in baseline)
    als=sum(r["sequential_120_success"]=="True" for r in baseline)
    uniform=sum(int(r["success"]) for r in pr)/100
    fixed=sum(int(r["success"]) for r in pr if int(r["pair_index"])==5)/10
    neural=sum(int(r["success"]) for r in cr)/10
    rates[split]=[ds/10,als/10,uniform,fixed,neural]
assert np.allclose(rates['validation'],[45.6,59,54.73,57.6,61.3])
assert np.allclose(rates['test'],[45.5,55.2,55.2,58.7,60.9])
audit["panda_success_rates_percent"]=rates

fig,ax=plt.subplots(figsize=(9.5,4.0))
y=np.arange(5)
labels=["Direct","All-link Sequential","Uniform selection","Preset pair (3,5)","Learned selection"]
for shift,split,color,label in [(-.18,'validation',BLUE,'Validation: 1,000 episodes'),(.18,'test',GREEN,'Test: 1,000 episodes')]:
    b=ax.barh(y+shift,rates[split],height=.31,color=color,label=label)
    ax.bar_label(b,labels=[f'{v:.2f}' for v in rates[split]],padding=5,fontsize=FONT_SIZE)
ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,74)
ax.set_xlabel("Success rate (%)"); decorate(ax)
ax.legend(frameon=False,ncol=2,fontsize=FONT_SIZE,loc='lower left',bbox_to_anchor=(0,1.02))
fig.tight_layout(); save(fig,"panda_success")

fig,ax=plt.subplots(figsize=(8.0,2.6))
values=rates['validation'][:2]
b=ax.barh(['Direct','All-link Sequential'],values,color=[GRAY,BLUE],height=.5)
ax.bar_label(b,labels=['45.60%  (456/1,000)','59.00%  (590/1,000)'],padding=6)
ax.invert_yaxis(); ax.set_xlim(0,86); ax.set_xlabel('Success rate (%)'); decorate(ax)
fig.tight_layout(); save(fig,'panda_all_links')

# Use paired intersections, never means over differently successful populations.
timing=[]
valbase={r['case_id']:r for r in all_rows}
# Keep data-field selection independent of presentation labels.
for label, field, selected in [
    ('Preset pair (3,5)', 'completion_time',
     [r for r in pair_rows if r['split']=='validation' and int(r['pair_index'])==5]),
    ('Learned selection', 'completion_time_seconds',
     [r for r in choices if r['split']=='validation' and r['family']=='baseline' and r['seed']=='11']),
]:
    shared=[r for r in selected if int(r['success']) and valbase[r['case_id']]['sequential_120_success']=='True']
    a=float(np.mean([float(valbase[r['case_id']]['sequential_120_completion_time']) for r in shared]))
    b=float(np.mean([float(r[field]) for r in shared]))
    timing.append({'candidate':label,'n':len(shared),'all_link_mean':a,'candidate_mean':b,'reduction_percent':100*(a-b)/a})
assert [t['n'] for t in timing]==[468,494]
fig,axes=plt.subplots(1,2,figsize=(8.7,3.6),sharex=True)
for ax,item,color in zip(axes,timing,[GRAY,GREEN]):
    vals=[item['all_link_mean'],item['candidate_mean']]
    b=ax.barh(['All-link\nSequential',item['candidate'].replace('Learned selection', 'Learned\nselection')],vals,color=[BLUE,color],height=.48)
    ax.bar_label(b,labels=[f'{x:.2f} s' for x in vals],padding=6)
    ax.invert_yaxis(); ax.set_xlim(0,63); ax.set_xlabel('Mean completion time (s)')
    ax.set_title(f"{item['n']} shared successes\n{item['reduction_percent']:.2f}% reduction",fontsize=FONT_SIZE,pad=14)
    decorate(ax)
fig.tight_layout(w_pad=2); save(fig,'paired_time'); audit['validation_paired_time']=timing

fig,ax=plt.subplots(figsize=(8,3.5))
pairs=summary['action_pairs']; vals=np.array(summary['splits']['validation']['pair_successes'])/10
colors=[GREEN if tuple(p)==(3,5) else BLUE for p in pairs]
b=ax.bar(np.arange(10),vals,color=colors,width=.9)
# White in-bar values stay clear of the 59% reference line.
ax.bar_label(b,labels=[f'{x:.1f}' for x in vals],label_type='center',
             fontsize=FONT_SIZE,color='white')
ax.axhline(59,color=RED,ls='--',lw=1,label='All-link Sequential: 59.0%')
ax.set_xticks(np.arange(10),[f'{p[0]},{p[1]}' for p in pairs]); ax.set_ylim(0,82)
ax.set_ylabel('Success rate (%)'); ax.set_xlabel('Selected intermediate link pair')
ax.legend(frameon=False,loc='upper right',fontsize=FONT_SIZE); fig.tight_layout(); save(fig,'pair_variability')

# Revision cycle 1: show actual stage changes and the orientation failure.
tr = traces['sequential']
changes = np.flatnonzero(np.diff(tr['active_link']) != 0) + 1
goal = np.searchsorted(tr['t'], case['sequential']['completion_time'])
# Goal positions are reconstructed from analytic IK and matched against every
# historical CLF sample. Never substitute the achieved final pose for q*.
reconstruction = read('planar_goal_reconstruction.json')
goals = np.array(reconstruction['lyapunov_reconstruction']['targets'])
np.testing.assert_allclose(fk(reconstruction['lyapunov_reconstruction']['selected_ik_q'])[1:], goals, atol=1e-12)
np.testing.assert_allclose(goals[-1], case['target'], atol=1e-12)
residuals = {}
for method, trm in traces.items():
    ep2 = ((trm['active_xy'] - goals[trm['active_link'].astype(int)-1])**2).sum(axis=1)
    v = 10*ep2 + 5*(1-np.cos(trm['final_yaw']-case['target_yaw']))*(trm['active_link']==3)
    residuals[method] = float(np.max(np.abs(v-trm['clf'])))
    np.testing.assert_allclose(v, trm['clf'], rtol=1e-11, atol=1e-10)
snapshot_indices = [changes[0]-1, changes[1]-1, goal]
snapshot_points = [fk(tr['q'][idx]) for idx in snapshot_indices]
spatial = [*snapshot_points, goals]
for obs in case['obstacles']:
    c=np.array(obs['position']); r=obs['radius']
    spatial.extend([np.array([c-r,c+r])])
spatial = np.concatenate(spatial)
lo,hi = spatial.min(axis=0)-.35,spatial.max(axis=0)+.35
assert np.all(spatial>lo) and np.all(spatial<hi)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharex=True, sharey=True)
for ax, idx, active, p in zip(axes, snapshot_indices, [1,2,3], snapshot_points):
    ik_pose=np.vstack(([0,0],goals))
    ax.plot(*ik_pose.T,'o--',color=GREEN,lw=2.6,ms=4,zorder=3)
    ax.plot(*p.T, 'o-', color='#a2aab2', lw=5, ms=6)
    ax.plot(*p[active-1:active+1].T, 'o-', color=BLUE, lw=6, ms=7)
    ax.plot(*p[active], 'o', ms=9, color=GREEN, zorder=5)
    target=goals[active-1]
    ax.plot(*np.array([p[active],target]).T, '--', color=GREEN,lw=2,zorder=6)
    ax.plot(*target, 'o', ms=13,markerfacecolor='none',markeredgecolor=GREEN,markeredgewidth=2,zorder=7)
    ax.annotate('$p_'+str(active)+'^*$',target,xytext=(7,-16),textcoords='offset points',fontsize=FONT_SIZE,color=GREEN)
    for obs in case['obstacles']:
        ax.add_patch(Circle(obs['position'], obs['radius'], color=RED, alpha=.5))
    ax.plot(0,0,'ks',ms=7)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal')
    ax.set_title(f"{'link '+str(active) if active < 3 else 'EE position + yaw'}\n{tr['t'][idx]:.2f} s",fontsize=FONT_SIZE)
    ax.axis('off')
fig.tight_layout(); save(fig,'planar_stages')
audit['planar_stage_snapshots_seconds'] = [float(tr['t'][i]) for i in snapshot_indices]
audit['planar_stage_geometry'] = {'lower_bounds':lo.tolist(),'upper_bounds':hi.tolist(),
    'all_points_and_obstacle_extents_inside':True,'goal_source':'Analytic IK reconstruction, verified against every stored CLF sample',
    'max_clf_residual':residuals,'goal_positions':goals.tolist()}
fig,ax = plt.subplots(figsize=(10,2.35))
for method,color,label in [('direct',GRAY,'Direct'),('sequential',BLUE,'Sequential')]:
    trm=traces[method]; angle=trm['final_yaw']-case['target_yaw']
    ax.plot(trm['t'][::15],np.abs(np.arctan2(np.sin(angle),np.cos(angle)))[::15],color=color,lw=2,label=label)
for x in tr['t'][changes]: ax.axvline(x,color=BLUE,alpha=.25,ls=':',lw=1)
ax.axhline(.1,color=RED,ls='--',lw=1)
ax.axvline(case['sequential']['completion_time'],color=GREEN,ls=':',lw=1.5)
ax.set(xlim=(0,35),ylabel='Yaw error (rad)',xlabel='Simulation time (s)')
ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=FONT_SIZE)
fig.tight_layout(); save(fig,'planar_yaw')

example=read('panda_example.json')
example_rows=[r for r in pair_rows if r['case_id']==example['case_id']]
assert [int(r['success']) for r in example_rows]==example['success']
fig=plt.figure(figsize=(6.0,5.1)); ax=fig.add_subplot(projection='3d')
phi,theta=np.meshgrid(np.linspace(0,2*np.pi,24),np.linspace(0,np.pi,16))
for x,y,z,r in example['obstacles']:
    ax.plot_surface(x+r*np.cos(phi)*np.sin(theta),y+r*np.sin(phi)*np.sin(theta),z+r*np.cos(theta),
                    color=RED,alpha=.17,linewidth=0,shade=False)
p=np.array(example['initial_frame_positions'])
ax.plot(*p.T,'o-',color=BLUE,lw=5,ms=5,label='Initial configuration')
target=np.array(example['target_position']); rot=np.array(example['target_rotation'])
ax.scatter(*target,color=GREEN,marker='*',s=150,depthshade=False,label='Goal pose')
for j,color in enumerate([RED,GREEN,BLUE]):
    ax.quiver(*target,*(.13*rot[:,j]),color=color,linewidth=2,arrow_length_ratio=.25)
ax.set(xlim=(-.95,.95),ylim=(-.95,.95),zlim=(0,1.55))
ax.set_box_aspect((1,1,.85));ax.view_init(elev=24,azim=35)
ax.set_xticks([-.5,0,.5]);ax.set_yticks([-.5,0,.5]);ax.set_zticks([0,.5,1])
ax.set_xlabel('x (m)',labelpad=1);ax.set_ylabel('y (m)',labelpad=1)
ax.tick_params(labelsize=FONT_SIZE,pad=0)
ax.legend(loc='upper left',fontsize=FONT_SIZE,frameon=False)
fig.subplots_adjust(left=0,right=1,bottom=.02,top=.98);save(fig,'panda_scene')
audit['panda_scene_example']={'case_id':example['case_id'],'obstacle_count':len(example['obstacles']),
    'pair_success_labels':example['success'],'rendering':'Static saved q0 and obstacles; robot drawn as frame connection lines, not collision geometry.'}

fig,ax=plt.subplots(figsize=(8.8,3.7))
vals=[rates['test'][i] for i in [0,1,3,4]]
b=ax.barh(['Direct','All-link Sequential','Preset pair (3,5)','Learned selection'],vals,
          color=[GRAY,BLUE,GRAY,GREEN],height=.54)
ax.bar_label(b,labels=[f'{v:.2f}%' for v in vals],padding=6,fontsize=FONT_SIZE)
ax.invert_yaxis();ax.set_xlim(0,77);ax.set_xlabel('Test success rate (%)');decorate(ax)
fig.tight_layout();save(fig,'panda_test_success')

# Revision cycle 2: use Test for both headline success and timing.
# Recompute each paired intersection from raw rows and verify archived means.
base={r['case_id']:r for r in test_rows}
fixed={r['case_id']:r for r in pair_rows if r['split']=='test' and int(r['pair_index'])==5}
learned={r['case_id']:r for r in choices if r['split']=='test' and r['family']=='baseline' and r['seed']=='11'}
test_timing=[]
archived=read('test_baselines_paired_comparisons.json')['paired_comparisons']
for key,label,reference,success_key,time_key in [
    ('neural_vs_sequential','All-link Sequential',base,'sequential_120_success','sequential_120_completion_time'),
    ('neural_vs_fixed35','Preset pair (3,5)',fixed,'success','completion_time')]:
    shared=[cid for cid,r in learned.items() if int(r['success']) and reference[cid][success_key] in ['True','1']]
    a=float(np.mean([float(reference[cid][time_key]) for cid in shared]))
    b=float(np.mean([float(learned[cid]['completion_time_seconds']) for cid in shared]))
    expected=archived[key]
    assert len(shared)==expected['both_success']
    np.testing.assert_allclose([a,b],[expected['reference_common_mean'],expected['candidate_common_mean']],atol=1e-10)
    test_timing.append({'reference':label,'n':len(shared),'reference_mean':a,'learned_mean':b,'difference_seconds':b-a})
fig,axes=plt.subplots(1,2,figsize=(9.3,3.7),sharex=True)
for ax,item in zip(axes,test_timing):
    vals=[item['reference_mean'],item['learned_mean']]
    bars=ax.barh([item['reference'].replace('All-link Sequential', 'All-link\nSequential'),'Learned\nselection'],vals,color=[BLUE,GREEN],height=.48)
    ax.bar_label(bars,labels=[f'{v:.2f} s' for v in vals],padding=6,fontsize=FONT_SIZE)
    ax.invert_yaxis();ax.set_xlim(0,65);ax.set_xlabel('Mean completion time (s)')
    change=f"{item['difference_seconds']:+.2f} s"
    ax.set_title(f"Test: {item['n']} shared successes\nDifference: {change}",fontsize=FONT_SIZE,pad=14)
    decorate(ax)
fig.tight_layout(w_pad=2);save(fig,'paired_time_test')
audit['test_paired_time']=test_timing

# Main-talk timing: one directly matched comparison, retained in full above for Q&A.
item = test_timing[0]
assert item['n'] == 467
assert [round(item[key], 2) for key in ['reference_mean', 'learned_mean']] == [47.97, 36.62]
fig, ax = plt.subplots(figsize=(8.0, 2.8))
vals = [item['reference_mean'], item['learned_mean']]
bars = ax.barh(['All-link Sequential', 'Learned selection'], vals,
               color=[BLUE, GREEN], height=.5)
ax.bar_label(bars, labels=[f'{value:.2f} s' for value in vals],
             padding=7, fontsize=FONT_SIZE)
ax.invert_yaxis()
ax.set(xlim=(0, 61), xticks=[0, 20, 40, 60], xlabel='Mean completion time (s)')
decorate(ax)
fig.tight_layout()
save(fig, 'paired_time_test_all_links')
audit['test_paired_time_main'] = {
    **item, 'split': 'test', 'evaluation_seconds': 120,
    'population': 'The same 467 episodes succeeded with both methods',
    'source_comparison_key': 'neural_vs_sequential',
    'appendix_comparison_retained': 'paired_time_test',
}


# Each planar task function evaluated along the SAME stored sequential motion.
# Inactive functions are evaluated for explanation; only one CLF is constrained.
seq=traces['sequential']; angles=np.cumsum(seq['q'],axis=1)
points=np.cumsum(np.stack([np.cos(angles),np.sin(angles)],axis=-1),axis=1)
values=10*np.sum((points-goals)**2,axis=-1)
values[:,-1]+=5*(1-np.cos(seq['final_yaw']-case['target_yaw']))
active=seq['active_link'].astype(int)
np.testing.assert_allclose(values[np.arange(len(active)),active-1],seq['clf'],atol=1e-10,rtol=1e-11)
fig,axes=plt.subplots(3,1,figsize=(9,4.8),sharex=True)
colors=[BLUE,GREEN,'#966233']
labels=[r'$V_1$',r'$V_2$',r'$V_{\mathrm{ee}}$']
stop=int(np.searchsorted(seq['t'],20))
for j,(ax,color,label) in enumerate(zip(axes,colors,labels)):
    mask=active==j+1
    ax.plot(seq['t'][:stop:10],values[:stop:10,j],color=color,lw=1.7,ls='--')
    indices=np.flatnonzero(mask & (seq['t']<=20))
    ax.plot(seq['t'][indices[::3]],values[indices[::3],j],color=color,lw=3.3)
    ax.axvspan(seq['t'][indices[0]],seq['t'][indices[-1]],color=color,alpha=.09)
    for x in seq['t'][changes]:ax.axvline(x,color='#74808a',lw=1,ls=':')
    ax.axvline(case['sequential']['completion_time'],color=GREEN,ls=':',lw=1)
    ax.set_ylabel(label,rotation=0,labelpad=23,fontsize=FONT_SIZE)
    ax.set_ylim(bottom=0);ax.set_xlim(0,20);ax.grid(axis='y',alpha=.15)
    ax.tick_params(labelsize=FONT_SIZE)
axes[-1].set_xlabel('Simulation time (s)')
axes[-1].set_xticks([0,5,10,15,20])
fig.tight_layout(h_pad=.35);save(fig,'planar_clf_time')
audit['planar_clf_time']={'function':'V1,V2=10||e_p||^2; Vee=10||e_p||^2+5(1-cos(e_yaw))',
    'active_samples_verified':len(active),'max_active_error':float(abs(values[np.arange(len(active)),active-1]-seq['clf']).max()),
    'time_range_seconds':[0,20],'first_goal_seconds':case['sequential']['completion_time'],
    'inactive_functions':'Re-evaluated on the same stored trajectory, not separate simulated runs.'}

# Qualitative function shapes with positive symbolic weights; the archived
# trajectory values retain their actual w_V=10 and w_psi=0.5 below.
shape_w_V, shape_w_psi = 1.0, 0.5
fig=plt.figure(figsize=(11.8,4.0))
e=np.linspace(-1,1,51);ex,ey=np.meshgrid(e,e)
for j in range(3):
    ax=fig.add_subplot(1,3,j+1,projection='3d')
    if j<2:
        x,y,z=ex,ey,shape_w_V*(ex**2+ey**2)
        xlabel,ylabel=r'$e_{'+str(j+1)+r',x}$ (m)',r'$e_{'+str(j+1)+r',y}$ (m)'
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([0,10,20])
    else:
        x,y=np.meshgrid(np.linspace(0,1,51),np.linspace(-np.pi,np.pi,65))
        z=shape_w_V*(x*x+shape_w_psi*(1-np.cos(y)))
        xlabel,ylabel=r'$\|e_p\|$ (m)',r'$e_\psi$ (rad)'
        ax.set_xticks([0,.5,1]);ax.set_yticks([-np.pi,0,np.pi]);ax.set_yticklabels([r'$-\pi$','0',r'$\pi$']);ax.set_zticks([0,10,20])
    ax.plot_surface(x,y,z,cmap='Blues' if j==0 else 'Greens' if j==1 else 'YlOrBr',alpha=.9,
                    linewidth=.2,edgecolor='white',rstride=3,cstride=3)
    ax.set_title(labels[j],fontsize=FONT_SIZE,pad=6)
    ax.set_xlabel(xlabel,labelpad=1,fontsize=FONT_SIZE);ax.set_ylabel(ylabel,labelpad=1,fontsize=FONT_SIZE)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    ax.tick_params(labelsize=FONT_SIZE,pad=0);ax.set_box_aspect((1,1,.78));ax.view_init(elev=27,azim=-55)
fig.subplots_adjust(left=0,right=.99,wspace=.12,bottom=.13,top=.9)
save(fig,'planar_clf_shapes')
audit['planar_clf_shapes']={'coordinates':'task error coordinates; Vee plotted versus position-error norm and yaw error',
    'symbolic_formula':'V_i=w_V||e_p,i||^2; Vee=w_V(||e_p||^2+w_psi(1-cos(e_psi)))',
    'illustrative_weights':{'w_V':shape_w_V,'w_psi':shape_w_psi},'archived_weights':{'w_V':10,'w_psi':0.5},
    'display':'qualitative shapes; vertical magnitude unlabelled; no changes to archived trajectory values'}


# Case 30: reference-style same-time comparison, reproduced from frozen records.
report30=read('case030_report.json'); case30=report30['original_case']
assert case30['case_id']==30 and case30['sequential']['success'] and not case30['direct']['success']
tr30={m:np.load(DATA/f'case_030_{m}.npz',allow_pickle=False) for m in ['direct','sequential']}
sq,di=tr30['sequential'],tr30['direct']
np.testing.assert_array_equal(sq['t'],di['t']);np.testing.assert_array_equal(sq['q'][0],di['q'][0])
t30=sq['t']; ac30=sq['active_link'].astype(int)
goals30=np.array(report30['lyapunov_reconstruction']['targets'])
np.testing.assert_allclose(fk(report30['lyapunov_reconstruction']['selected_ik_q'])[1:],goals30,atol=1e-12)
points30=np.array([fk(q)[1:] for q in sq['q']])
np.testing.assert_allclose(points30[:,-1],sq['final_xy'],atol=1e-12)
w_v30=2*report30['lyapunov_reconstruction']['clf_weight']
w_psi30=report30['lyapunov_reconstruction']['final_yaw_weight']
v30=w_v30*np.sum((points30-goals30)**2,axis=-1)
v30[:,-1]+=w_v30*w_psi30*(1-np.cos(sq['final_yaw']-case30['target_yaw']))
direct30=w_v30*(np.sum((di['final_xy']-case30['target'])**2,axis=1)+w_psi30*(1-np.cos(di['final_yaw']-case30['target_yaw'])))
np.testing.assert_allclose(v30[np.arange(len(t30)),ac30-1],sq['clf'],rtol=1e-11,atol=1e-10)
np.testing.assert_allclose(direct30,di['clf'],rtol=1e-11,atol=1e-10)
snap30=report30['snapshots'];indices30=[x['sample_index'] for x in snap30]
np.testing.assert_array_equal(np.flatnonzero(np.diff(ac30)!=0)+1,indices30[:2])
np.testing.assert_allclose(t30[indices30],[2.128,6.874,18.226],atol=1e-9)
for snap in snap30:np.testing.assert_allclose(sq['q'][snap['sample_index']],snap['q'],atol=1e-12)
end30=indices30[-1];tend30=float(t30[end30])
end_error=np.linalg.norm(sq['final_xy']-case30['target'],axis=1)
yaw_error=np.abs(np.arctan2(np.sin(sq['final_yaw']-case30['target_yaw']),np.cos(sq['final_yaw']-case30['target_yaw'])))
assert np.flatnonzero((end_error<=.1)&(yaw_error<=.1)&(ac30==3))[0]==end30
states30=[sq['q'][0]]+[sq['q'][i] for i in indices30]+[di['q'][end30]]
times30=[0.,*list(t30[indices30]),tend30]
active30=[None,1,2,3,3]
linkcolors30=['#0072B2','#E69F00','#009E73'];directcolor30='#A74484'
# Match the reference's documented nominal obstacle halo; this is not an exact RDF contour.
margin30=.08
cloud30=[np.zeros((1,2)),goals30,*[fk(q) for q in states30]]
for obs in case30['obstacles']:
    c=np.array(obs['position']);r=obs['radius']+margin30
    cloud30.append(np.array([c-r,c+r]))
cloud30=np.concatenate(cloud30);lo30=cloud30.min(axis=0)-.22;hi30=cloud30.max(axis=0)+.22
assert np.all(cloud30>lo30) and np.all(cloud30<hi30)

def scene30(ax,q,active):
    points=fk(q);ik=np.vstack(([0,0],goals30))
    ax.plot(*ik.T,color='#8a8f95',ls='--',lw=2,zorder=1)
    for obs in case30['obstacles']:
        ax.add_patch(Circle(obs['position'],obs['radius']+margin30,fill=False,edgecolor='#9b9b9b',ls=':',lw=1.5,zorder=1))
        ax.add_patch(Circle(obs['position'],obs['radius'],facecolor='#d0d0d0',edgecolor='#666666',lw=1.5,zorder=2))
    for j,color in enumerate(linkcolors30):
        a,b=points[j:j+2];delta=b-a;normal=.05*np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
        ax.add_patch(Polygon([a+normal,b+normal,b-normal,a-normal],color=color,zorder=3))
        for endpoint in [a,b]:ax.add_patch(Circle(endpoint,.05,color=color,zorder=3))
        ax.plot(*goals30[j],marker='*',ms=10,color=color,markeredgecolor='white',markeredgewidth=.5,zorder=6)
    ax.plot(*points.T,'o',ms=4,mfc='white',mec='#444444',mew=.7,zorder=5)
    ax.plot(0,0,'s',ms=7,color='#333333',zorder=6)
    if active is not None:ax.plot(*points[active],'o',ms=13,mfc='none',mec='#333333',mew=1.4,zorder=7)
    for point,yaw,color in [(goals30[-1],case30['target_yaw'],'#555555')]+([(points[-1],sum(q),linkcolors30[-1])] if active==3 else []):
        ax.annotate('',xy=point+.25*np.array([np.cos(yaw),np.sin(yaw)]),xytext=point,
                    arrowprops={'arrowstyle':'->','color':color,'lw':1.3,'mutation_scale':9})
    ax.set(xlim=(lo30[0],hi30[0]),ylim=(lo30[1],hi30[1]),aspect='equal');ax.axis('off')

pose_titles30=['Initial configuration','Sequential: link 1','Sequential: link 2','Sequential: EE','Direct: EE']
legend30=[Patch(facecolor='#d0d0d0',edgecolor='#666666',label='Obstacle'),
          Line2D([],[],color='#9b9b9b',ls=':',label='Clearance margin'),
          Line2D([],[],color='#8a8f95',ls='--',label='IK configuration'),
          Line2D([],[],color='#444444',marker='*',ls='none',ms=9,label='Link targets'),
          Line2D([],[],color='#444444',marker='o',mfc='none',ls='none',ms=9,label='Control point')]
fig,axes=plt.subplots(1,5,figsize=(12.2,4.5))
for j,(ax,q,active,title,t) in enumerate(zip(axes,states30,active30,pose_titles30,times30)):
    scene30(ax,q,active)
    ax.set_title(f'{title}\n$t={t:.3f}$ s',fontsize=FONT_SIZE,pad=9,color=directcolor30 if j==4 else '#243447')
fig.subplots_adjust(left=.005,right=.995,top=.82,bottom=.12,wspace=.1)
fig.legend(handles=legend30,loc='lower center',ncol=5,fontsize=FONT_SIZE,frameon=False,handlelength=1.6,columnspacing=1.1)
save(fig,'case030_stages')

def clf30(ax,j,title=True):
    color=linkcolors30[j]
    # All stored samples retained; no smoothing or new rollout.
    ax.plot(t30,v30[:,j],color=color,ls='--',lw=1.5)
    ax.plot(t30,np.where(ac30==j+1,v30[:,j],np.nan),color=color,lw=2.8,label='Sequential')
    if j<2:
        start=0 if j==0 else t30[indices30[0]]
        ax.axvspan(start,t30[indices30[j]],color=color,alpha=.07)
    else:
        ax.plot(t30,direct30,color=directcolor30,ls='-.',lw=2.3,label='Direct')
        ax.axvline(tend30,color='#444444',ls='-.',lw=1)
        ax.legend(loc='upper right',frameon=False,fontsize=FONT_SIZE,
                  handlelength=.9,handletextpad=.4,labelspacing=.15,borderaxespad=.1)
    for x in t30[indices30[:2]]:ax.axvline(x,color='#999999',ls=':',lw=1)
    ax.set_xlim(0,35);ax.set_ylim(bottom=0);ax.set_xticks([0,10,20,35]);ax.grid(axis='y',alpha=.17)
    if j == 2:
        ax.set_ylim(0, 210)
        ax.set_yticks([0, 100, 200])
    ax.set_xlabel('Time (s)',fontsize=FONT_SIZE);ax.tick_params(labelsize=FONT_SIZE)
    if title:ax.set_title([r'Sequential: $V_1$',r'Sequential: $V_2$',r'Common EE CLF $V_{\mathrm{ee}}$'][j],fontsize=FONT_SIZE,pad=10)

fig,axes=plt.subplots(1,3,figsize=(10.5,3.25))
for j,ax in enumerate(axes):clf30(ax,j)
fig.tight_layout(w_pad=1.3);save(fig,'case030_clf_time')
# Full reference arrangement for Q&A / export: five poses above three CLF plots.
fig=plt.figure(figsize=(12.2,7.7));gs=fig.add_gridspec(3,1,height_ratios=[1.15,.18,1],hspace=.14)
upper=gs[0].subgridspec(1,5,wspace=.1)
for j,(q,active,title,t) in enumerate(zip(states30,active30,pose_titles30,times30)):
    ax=fig.add_subplot(upper[j]);scene30(ax,q,active)
    ax.set_title(f'{title}\n$t={t:.3f}$ s',fontsize=FONT_SIZE,pad=5,color=directcolor30 if j==4 else '#243447')
key=fig.add_subplot(gs[1]);key.axis('off');key.legend(handles=legend30,loc='center',ncol=5,frameon=False,fontsize=FONT_SIZE)
lower=gs[2].subgridspec(1,3,wspace=.28)
for j in range(3):clf30(fig.add_subplot(lower[j]),j)
fig.subplots_adjust(left=.045,right=.995,top=.94,bottom=.09)
save(fig,'case030_overview')
audit['case30']={'case_id':30,'snapshots_seconds':times30,'shared_last_snapshot_time':tend30,
    'same_initial_state':True,'sequential_success':True,'direct_success':False,'direct_failure_class':case30['direct']['failure_reason'],
    'direct_qp_infeasible_count':case30['direct']['qp_infeasible_count'],
    'direct_same_time_position_error':float(np.linalg.norm(di['final_xy'][end30]-case30['target'])),
    'direct_same_time_yaw_error':float(abs(np.arctan2(np.sin(di['final_yaw'][end30]-case30['target_yaw']),np.cos(di['final_yaw'][end30]-case30['target_yaw'])))),
    'raw_samples_per_method':len(t30),'max_active_clf_error':float(np.max(abs(v30[np.arange(len(t30)),ac30-1]-sq['clf']))),
    'max_direct_clf_error':float(np.max(abs(direct30-di['clf']))),
    'actual_weights':{'w_V':w_v30,'w_psi':w_psi30},'all_geometry_inside_common_axes':True,
    'obstacle_halo':'Nominal geometric 0.08 m margin, matching archived reference; not exact RDF contours',
    'figure_files':['case030_stages','case030_clf_time','case030_overview'],'no_new_simulation':True}
for item in tr30.values():item.close()

for t in traces.values(): t.close()
assert requested_figures <= set(generated_figures), f'Unknown figure stems: {requested_figures - set(generated_figures)}'
if requested_figures:
    # A presentation-only partial render leaves the complete data audit intact.
    print(json.dumps({'regenerated_figures': generated_figures, 'all_data_checks_passed': True}, indent=2))
else:
    (DATA/'figure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

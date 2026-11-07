from pathlib import Path
exec(Path('outputs/linear-insertion-aligned-20261107/final/solve_alignment.py').read_text().split('results=[]')[0])
audit=NativeBoundaryAudit(m,max_samples=4096);gcase=m.geom('servo_3_case').id;gupper=m.geom('upper_arm_module_printed_0').id;rows=[]
for dx in np.arange(-.00215,-.002701,-.00005):
 state(0);d.qpos[qa:qa+3]+=rr@np.array([dx,0,0]);mujoco.mj_forward(m,d);bounds=audit._bounds(d,[gcase,gupper]);pair=audit._pairs(d,[(gcase,gupper)],bounds)[0];row={'dx':dx,'max_m':pair['penetration_m'],'case_in_pocket':pair['a_in_b_m'],'pocket_in_case':pair['b_in_a_m']};rows.append(row);print(row,flush=True)
(out/'pocket-fit.json').write_text(json.dumps(rows,indent=2)+'\n')

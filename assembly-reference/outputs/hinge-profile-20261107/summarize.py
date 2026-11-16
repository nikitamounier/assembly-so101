from pathlib import Path
import collections,json,statistics
h=Path(__file__).resolve().parent
versions={v:json.loads((h/v/'panel.json').read_text()) for v in ['v10','v12','v13']}
out={}
for v,x in versions.items():
 groups=collections.defaultdict(list)
 for r in x['rows']:
  if not r['warmup']:groups[r['stage']].append(r)
 out[v]=dict(signature=x['signature'],hardware=x['hardware'],npair=x['npair'],nstate=x['nstate'],stages={k:dict(wall_s=statistics.median(r['wall_s'] for r in rs),rollout_s=statistics.median(r['spans']['rollout'] for r in rs),refresh_s=statistics.median(r['spans']['refresh'] for r in rs),reset_s=statistics.median(r['reset_s'] for r in rs),cpu_s=statistics.median(r['cpu_s'] for r in rs),wall_samples=[r['wall_s'] for r in rs]) for k,rs in groups.items()})
 out[v]['sum_stage_medians_s']=sum(z['wall_s'] for z in out[v]['stages'].values())
for v in ['v12','v13']:
 out[v]['speedup_vs_v10']={k:out['v10']['stages'][k]['wall_s']/z['wall_s'] for k,z in out[v]['stages'].items()}
 out[v]['aggregate_speedup_vs_v10']=out['v10']['sum_stage_medians_s']/out[v]['sum_stage_medians_s']
out['v13']['speedup_vs_v12']={k:out['v12']['stages'][k]['wall_s']/z['wall_s'] for k,z in out['v13']['stages'].items()}
for v in ['v10','v12','v13']:
 p=h/v/'ppo-result.json'
 if p.exists():out[v]['ppo']=json.loads(p.read_text())
 p=h/(v+'-timed')/'panel.json'
 if p.exists():out[v]['native_timed']=json.loads(p.read_text())
for v in ['v12','v13']:
 p=h/(v+'-repeat')/'ppo-result.json'
 if p.exists():
  out[v]['ppo_repeat']=json.loads(p.read_text());out[v]['ppo_mean_elapsed_s']=statistics.mean([out[v]['ppo']['elapsed_s'],out[v]['ppo_repeat']['elapsed_s']])
(h/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({v:{k:x[k] for k in ['sum_stage_medians_s','aggregate_speedup_vs_v10'] if k in x} for v,x in out.items()},indent=2))

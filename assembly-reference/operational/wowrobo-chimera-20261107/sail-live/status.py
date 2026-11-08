"""Fresh provider/process evidence; cumulative readiness is labeled separately."""
import json, os, time
from pathlib import Path
import sail
from launch import retry_io

HERE=Path(__file__).resolve().parent
for line in (HERE.parents[3]/'.env').read_text().splitlines():
    if line.startswith('SAIL_API_KEY='):os.environ['SAIL_API_KEY']=line.split('=',1)[1].strip().strip('"').strip("'")
receipt=json.loads((HERE/'active.json').read_text())
rows=[]
probe='''import json,psutil
from pathlib import Path
processes=list(psutil.process_iter(['cmdline']))
names=[p.info['cmdline'] or [] for p in processes]
live={p.pid for p in processes if '/opt/sail-live/worker.py' in (p.info['cmdline'] or [])}
lines=Path('/run/assembly/workers.log').read_text()
events=[]
for line in lines.splitlines():
 try:events.append(json.loads(line))
 except json.JSONDecodeError:pass
finite={e['pid'] for e in events if e.get('event')=='first_control' and e.get('finite')}
print(json.dumps(dict(worker_processes=len(live),supervisors=sum('/opt/sail-live/supervise.py' in c for c in names),live_workers_with_finite_first_control=len(live&finite),cumulative_initialized=lines.count('"event": "initialized"'),cumulative_first_controls=lines.count('"event": "first_control"'))))
'''
for box_id in receipt.get('boxes',[]):
    box=retry_io(lambda: sail.Sailbox.get(box_id))
    row=dict(box=box_id,role='worker' if box_id in receipt.get('worker_boxes',[]) else 'broker',status=box.status,
             reserved_vcpus=box.vcpu_count,used_vcpus=box.cpu_used_vcpu,memory_used_bytes=box.memory_used_bytes)
    if row['role']=='worker' and box.status=='running':
        result=retry_io(lambda: box.exec(['python','-c',probe],timeout=30).wait())
        if result.exit_code:row['probe_error']=result.stderr[-500:]
        else:row.update(json.loads(result.stdout))
    rows.append(row)
report=dict(time_unix=time.time(),pool_status=receipt['status'],app_id=receipt.get('app_id'),boxes=rows,
            running_total_reserved_vcpus=sum((r['reserved_vcpus'] or 0) for r in rows if r['status']=='running'),
            running_worker_reserved_vcpus=sum((r['reserved_vcpus'] or 0) for r in rows if r['role']=='worker' and r['status']=='running'),
            live_worker_processes=sum(r.get('worker_processes',0) for r in rows),
            live_workers_with_finite_first_control=sum(r.get('live_workers_with_finite_first_control',0) for r in rows))
print(json.dumps(report))

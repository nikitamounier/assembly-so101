"""CPU observer benchmark; no training, GPU, rendering or physics is executed."""
import importlib.util,json,time
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parent.parent/'assembly-so101/assembly_policy/wowrobo/telemetry.py'
spec=importlib.util.spec_from_file_location('bounded_replay_benchmark',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Sink:
    def submit(self,clip):return True
    def close(self):pass

N,B,G=5090,96,29
entry_meta=[dict(task='task'+str(i//7),stage='start'+str(i)) for i in range(G)]
rec=module.EpisodeRecorder('/fixture',entry_meta,['body'+str(i) for i in range(B)],N,{},writer=Sink(),refresh_seconds=0)
obs=dict(part_p=np.zeros((N,B,3),np.float32),part_q=np.tile(np.array([1,0,0,0],np.float32),(N,B,1)))
entries=np.arange(N)%G;valid=np.ones(N,bool);reward=np.zeros(N,np.float32)
durations=[]
for step in range(400):
    done=np.full(N,step%40==39);data=dict(valid=valid,success=np.zeros(N,bool),safety=done,timeout=np.zeros(N,bool))
    start=time.perf_counter();rec.before_step(obs,entries,valid,now=step);rec.after_step(reward,done,data,obs if done.any() else None,step,now=step);durations.append(time.perf_counter()-start)
result=dict(envs=N,bodies=B,task_start_groups=G,measured_steps=len(durations),mean_ms=float(np.mean(durations)*1000),p99_ms=float(np.percentile(durations,99)*1000),max_ms=float(np.max(durations)*1000),frame_buffers_mib=(rec.frames.nbytes+rec.times.nbytes)/2**20,scope='observer CPU work including bounded clip copies; disk compression occurs on the separate bounded writer')
(Path(__file__).parent/'replay-overhead.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

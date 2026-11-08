"""Run inside spare allocation3606264; terminate only processes started here."""
from common import *
import subprocess
import shutil
import numpy as np

assert os.environ['SLURM_JOB_ID']=='3606264'
OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'active-inputs-v9.json').read_text())
def verify():
    bad=[n for n,h in manifest.items() if sha(ROOT/n)!=h];assert not bad,bad
    assert sha(ROOT/'assembly-so101/plugin/so101_surface_tree.h')=='3df103be0e3b8cfde509a00da787ad753fc40cd4686db9a4b45163be70e88d8f'
verify()
source=ROOT/'run-v6/checkpoint.before-bare-v8.pt'
assert sha(source)=='df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b'
write(OUT/'identity.json',dict(verified_files=len(manifest),source_manifest=manifest,checkpoint_sha256=sha(source),hardware=hardware(),start_unix=time.time(),source_root=str(ROOT)))
def call(script,*args):subprocess.run([sys.executable,str(OUT/script),*args],check=True)
call('instrument.py')
call('probe.py','baseline')
call('probe.py','instrumented')
a=np.load(OUT/'baseline-parity.npz');b=np.load(OUT/'instrumented-parity.npz')
checks={k:np.array_equal(a[k],b[k],equal_nan=True) for k in a.files}
assert a.files==b.files and all(checks.values()),checks
write(OUT/'parity.json',dict(status='PASS',arrays=len(checks),checks=checks,scope='Four complete160-substep trajectories, sensors, final state, observations, reward/done/info, contacts and forces; instrumented versus original native.'))
run=OUT/'ppo';run.mkdir(exist_ok=True);shutil.copy2(source,run/'checkpoint.pt')
hw=hardware();representatives={}
for cpu in hw['affinity']:
    p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology');key=((p/'physical_package_id').read_text(),(p/'core_id').read_text());representatives.setdefault(key,cpu)
cpus=list(representatives.values());assert len(cpus)>=3,hw
children=[];logs=[]
try:
    for i in range(2):
        log=(OUT/f'worker-{i}.log').open('w');logs.append(log)
        children.append(subprocess.Popen(['taskset','-c',str(cpus[i]),sys.executable,str(OUT/'capture.py'),'worker',str(i)],stdout=log,stderr=subprocess.STDOUT))
    learner_log=(OUT/'learner.log').open('w');logs.append(learner_log)
    learner=subprocess.Popen(['taskset','-c',str(cpus[2]),sys.executable,str(OUT/'capture.py'),'learner'],stdout=learner_log,stderr=subprocess.STDOUT);children.append(learner)
    learner.wait(timeout=1500)
    assert learner.returncode==0,f'Learner failed: {learner.returncode}'
    for child in children[:2]:child.wait(timeout=30);assert child.returncode==0
finally:
    for child in children:
        if child.poll() is None:child.terminate()
    for child in children:
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    for log in logs:log.close()
verify()
write(OUT/'complete.json',dict(status='PASS',end_unix=time.time(),all_children_closed=True,baseline_manifest_verified_after=True,checkpoint_sha256=sha(run/'checkpoint.pt')))

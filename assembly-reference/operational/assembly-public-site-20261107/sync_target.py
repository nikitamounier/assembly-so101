"""Keep the one public site pointed at the latest verified display publication."""
import argparse,json,time,urllib.request,urllib.parse,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--preview-status',type=Path,required=True);a=p.parse_args();workspace=a.workspace.resolve();project=workspace/'robotics-assembly-3.0';target=workspace/'assembly-dashboard/state-viewer-target.json';full=project/'outputs/reset-bank-19200-completion-20261107/status.json';borrowed=project/'outputs/reset-bank-19200-borrowed-training/status.json';out=project/'operational/assembly-public-site-20261107/target-sync-status.json'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def atom(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
last_identity=None
while True:
 try:
  f=read(full);v=read(a.preview_status);chosen=None;is_full=False
  if f.get('status')=='complete' and f.get('viewer_url'):chosen=f;is_full=True
  elif v.get('viewer_url') and v.get('accepted',v.get('accepted_variations',0))>0:chosen=v
  if chosen:
   url=chosen['viewer_url'];u=urllib.parse.urlsplit(url)
   assert u.scheme=='http' and u.hostname in ('localhost','127.0.0.1') and u.port
   accepted=19200 if is_full else chosen.get('accepted',chosen.get('accepted_variations'))
   identity=(url,accepted,is_full)
   if identity!=last_identity:
    with urllib.request.urlopen(url.rstrip('/')+'/api/catalog',timeout=20) as response:catalog=json.load(response)
    variants=[e for e in catalog['entries'] if e.get('variation',{}).get('kind')=='variation'];assert len(variants)==accepted and all(e['variation']['training_eligible'] for e in variants)
    by_split=chosen.get('counts',{});train=chosen.get('accepted_train',by_split.get('train'));evaluation=chosen.get('accepted_eval',by_split.get('eval'))
    if is_full:train,evaluation=16000,3200
    if train is None or evaluation is None:raise ValueError('Publication receipt needs split counts')
    assert train+evaluation==accepted
    current=read(target);stamp=time.time();new=dict(url='http://127.0.0.1:'+str(u.port),bank_status='complete' if is_full else 'building',accepted_train=train,accepted_eval=evaluation,target_train=16000,target_eval=3200,updated_at=stamp,snapshot_label='Complete19,200-state validated bank' if is_full else f'Published snapshot · {accepted:,} accepted variations · generation continues',training=current.get('training',dict(state='paused',checkpoint=32,reason='Training resources are preparing the19,200-state bank')))
    atom(target,new);last_identity=identity
   current=read(target);state=read(borrowed).get('status','')
   if state=='state_build_complete_training_cpu_configuration_restored' and current.get('training',{}).get('state')=='paused':
    current['training']=dict(state='restoring',checkpoint=32,reason='State bank complete. Training CPU configuration restored; the live dashboard reports actual resumed updates.');atom(target,current)
   atom(out,dict(status='serving_verified_publication',accepted=accepted,complete=is_full,viewer_url=url,pid=os.getpid(),checked_unix=time.time()))
  time.sleep(20)
 except Exception as error:
  atom(out,dict(status='retrying_publication_check',error=str(error),pid=os.getpid(),checked_unix=time.time()));time.sleep(20)

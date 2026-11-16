from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent.parent;RUN=ROOT/'run-v8';REL=ROOT/'bare-v8'
assert json.loads((REL/'migration.json').read_text())['status']=='PASS'
s=json.loads((RUN/'server.json').read_text());assert s['n_conns']==960
alias=ROOT/'v8'
if alias.exists():assert alias.resolve()==ROOT/'ops-v8'
else:alias.symlink_to(ROOT/'ops-v8',target_is_directory=True)
(ROOT/'migration-v8.json').write_bytes((REL/'migration.json').read_bytes())
print('run-v8 dashboard route ready')

"""Extract launch geometry from the captured actual CUDA graph executions."""
import json
import sqlite3
import sys
from pathlib import Path

root=Path(sys.argv[1])
result={}
for filename in root.glob('*nsys-trace-fixed/trace.sqlite'):
    db=sqlite3.connect(filename);db.row_factory=sqlite3.Row
    columns=[r[1] for r in db.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')]
    names='demangledName' if 'demangledName' in columns else 'shortName'
    registers='k.registersPerThread' if 'registersPerThread' in columns else '0'
    rows=db.execute(f'''select s.value name,k.gridX grid,k.blockX block,{registers} registers,
                      count(*) launches,sum(k.end-k.start)/1e6 gpu_ms
                      from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.{names}=s.id
                      where s.value like ? group by s.value,k.gridX,k.blockX,{registers}''',('%_noslip_solve%',)).fetchall()
    totals=db.execute('select count(*),sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
    result[filename.parent.name]={'noslip': [dict(x) for x in rows],'kernel_launches': totals[0],'kernel_ms': totals[1]}
    db.close()
print(json.dumps(result,indent=2))

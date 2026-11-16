"""Read-only health check for the approved local dashboard and current public tunnel."""
from pathlib import Path
import gzip,json,time,urllib.request,urllib.error,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'wowrobo-chimera-20261107/modal-live'))
from receipt_preflight import check
check(HERE)
site=json.loads((HERE/'public-site.json').read_text())
results={}
for name,url in [('local','http://127.0.0.1:8881/api/live'),('public',site['url'].rstrip('/')+'/api/live')]:
 try:
  request=urllib.request.Request(url,headers={'Accept-Encoding':'gzip'})
  with urllib.request.urlopen(request,timeout=15) as response:
   body=response.read()
   if response.headers.get('Content-Encoding','').lower()=='gzip':
    body=gzip.decompress(body)
   payload=json.loads(body);latest=(payload.get('history') or [{}])[-1]
   results[name]=dict(http=response.status,version=payload.get('version'),iteration=latest.get('it'),connection=payload.get('connection'))
 except Exception as error:results[name]=dict(error=type(error).__name__+': '+str(error))
report=dict(time=time.time(),url=site['url'],origin='http://127.0.0.1:8881',action='keep_monitor' if all(x.get('http')==200 for x in results.values()) else 'dashboard_needs_attention',checks=results,read_only=True)
p=HERE/'site-health-latest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p);print(json.dumps(report))

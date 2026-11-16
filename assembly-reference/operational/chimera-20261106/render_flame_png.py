import sys,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from render_flames import tree
for arg in sys.argv[1:]:
 p=Path(arg);d=json.loads(p.read_text());root=tree(d);layout=[]
 def walk(n,x,w,depth):
  if w<root['value']*.0002:return
  layout.append((n,x,w,depth));offset=x
  for child in n['children']:
   cw=w*child['value']/n['value'];walk(child,offset,cw,depth+1);offset+=cw
 walk(root,0,root['value'],0);maxdepth=max(v[3] for v in layout)
 fig,ax=plt.subplots(figsize=(16,max(3.5,(maxdepth+1)*.30+1.1)),layout='constrained')
 for node,x,w,depth in layout:
  h=int(hashlib.md5(node['name'].encode()).hexdigest()[:4],16)
  color=plt.cm.YlOrRd(.22+(h%500)/1000)
  ax.add_patch(Rectangle((x,depth),w,.94,facecolor=color,edgecolor='white',lw=.4))
  count=int(w/root['value']*225)
  if count>7:
   label=node['name'].split(' (')[0]
   ax.text(x+root['value']*.002,depth+.48,label[:count],va='center',fontsize=9,clip_on=True)
 ax.set(xlim=(0,root['value']),ylim=(maxdepth+1,-.5),yticks=[],xlabel='Width = wall-stack or native-phase duration; simultaneous worker/thread times are summed')
 ax.set_title(d['profiles'][0]['name'],loc='left',fontweight='bold',fontsize=15,pad=15)
 for spine in ax.spines.values():spine.set_visible(False)
 fig.savefig(p.with_suffix('.png'),dpi=150);plt.close(fig)

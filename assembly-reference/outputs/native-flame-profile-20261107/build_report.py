"""Self-contained, zoomable flame viewer. Native widths are measured timers."""
from pathlib import Path
import json
import html

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def root(name):return dict(name=name,value=0,children={})
def add(tree,stack,weight):
    tree['value']+=weight
    for name in stack:
        tree=tree['children'].setdefault(name,root(name));tree['value']+=weight
def pack(tree):return dict(name=tree['name'],value=tree['value'],children=[pack(x) for x in sorted(tree['children'].values(),key=lambda x:-x['value'])])
views=[]
for file in ['learner.speedscope.json','worker-0.speedscope.json','worker-1.speedscope.json']:
    data=read(file);p=data['profiles'][0];tree=root(file.split('.')[0]);frames=data['shared']['frames']
    for sample,weight in zip(p['samples'],p['weights']):add(tree,[frames[i]['name'] for i in sample],weight)
    views.append(dict(title=file.split('.')[0]+' — sampled Python wall stacks',note=p['name']+'; native C++ functions are opaque in these samples. Worker lifetime includes initialization and barrier waits; learner covers collection through checkpoint. Concurrent views must not be summed.',tree=pack(tree)))
native=read('instrumented-probe.json');baseline=read('baseline-probe.json');result=read('ppo-result.json');parity=read('parity.json');instrument=read('instrumentation.json')
native_tree=root('Four native rollout probes (timer hierarchy)')
for row in native['rows']:
    case=f"{row['stage']} / {'variation' if row['variation'] else 'prototype'}"
    attributed=0
    for pair in row['pairs']:
        s=pair['stats'];pair_name=' ↔ '.join(pair['names'])
        for name,w in [('Original SDF search',s[1]),('Surface supplement',s[2])]:add(native_tree,[case,pair_name,name],w);attributed+=w
    add(native_tree,[case,'Other rollout work / timer residual'],max(0,row['rollout_s']-attributed))
views.append(dict(title='Native pairs — measured timer flame hierarchy',note='Widths are actual pair timers from four isolated zero-action160-substep intervals, not unwound native stack samples or production-wide proportions. Original SDF and supplement are timed separately. No native function stack is invented. Pair labels retain exact geometry names.',tree=pack(native_tree)))
if (HERE/'instrumented-timed-probe.json').exists():
    phase=root('Native engine phases + collision pairs (separate phase-clock pass)')
    hierarchy={'mjTIMER_STEP':['mjTIMER_FORWARD','mjTIMER_ADVANCE'],'mjTIMER_FORWARD':['mjTIMER_POSITION','mjTIMER_VELOCITY','mjTIMER_ACTUATION','mjTIMER_CONSTRAINT'],'mjTIMER_POSITION':['mjTIMER_POS_KINEMATICS','mjTIMER_POS_INERTIA','mjTIMER_POS_COLLISION','mjTIMER_POS_MAKE','mjTIMER_POS_PROJECT'],'mjTIMER_POS_COLLISION':['mjTIMER_COL_BROAD','mjTIMER_COL_NARROW']}
    for row in read('instrumented-timed-probe.json')['rows']:
        timers=row['native_timers'];case=row['stage']+(' variation' if row['variation'] else ' prototype')
        def insert(name,path):
            children=hierarchy.get(name,[]);value=timers[name]['seconds']
            if name=='mjTIMER_COL_NARROW':
                measured=0
                for pair in row['pairs']:
                    for title,w in [('Original SDF search',pair['stats'][1]),('Surface supplement',pair['stats'][2])]:
                        add(phase,path+[name,' ↔ '.join(pair['names']),title],w);measured+=w
                assert measured<=value+1e-6
                add(phase,path+[name,'Other narrow phase / timer overhead'],max(0,value-measured))
            else:
                for child in children:insert(child,path+[name])
                exclusive=value-sum(timers[c]['seconds'] for c in children)
                assert exclusive>=-1e-6,(name,exclusive)
                add(phase,path+[name]+(['Other '+name] if children else []),max(0,exclusive))
        insert('mjTIMER_STEP',[case])
        add(phase,[case,'Rollout wrapper residual'],max(0,row['rollout_s']-timers['mjTIMER_STEP']['seconds']))
    views.append(dict(title='Native engine → narrow phase → pairs',note='Actual nested MuJoCo timers from a separate native-clock pass, with same-pass original SDF/supplement pair attribution. Inclusive timers are converted to exclusive children and residuals to avoid double counting. All160substeps per start;72arrays remain byte-identical. Timer hierarchy, not unwound C call stacks.',tree=pack(phase)))
rows=[]
for row in native['rows']:
    for pair in row['pairs']:
        s=pair['stats'];rows.append({'start':row['stage']+(' variation' if row['variation'] else ''),'pair':' ↔ '.join(pair['names']),'bodies':' ↔ '.join(pair['bodies']),'role':pair['role'],**dict(zip(instrument['columns'],s))})
rows.sort(key=lambda r:-(r['sdf_s']+r['supplement_s']))
payload=json.dumps(dict(views=views,rows=rows,result=result,baseline=baseline,native=native,parity=parity)).replace('</','<\/')
page='''<!doctype html><html><head><meta charset="utf-8"><title>Frozen WowRobo native flame profile</title><style>
body{font:15px system-ui;background:#0d1421;color:#e0e8f2;margin:0;padding:28px}h1{font-size:25px}h2{font-size:19px;margin-top:30px}.sub,#note{color:#a9bad0;max-width:1100px;line-height:1.6}select,input,button{background:#1b2a40;color:#e0e8f2;border:1px solid #41516b;padding:8px;border-radius:6px;margin:8px 7px 10px 0}#graph{width:100%;overflow:auto;background:#121e30;border-radius:8px}svg text{fill:#111;font:12px system-ui;pointer-events:none}svg rect{stroke:#111;stroke-width:.5;cursor:pointer}#detail{min-height:40px;white-space:pre-wrap;overflow-wrap:anywhere;color:#d5e1f0;padding:12px 0}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px;text-align:right;border-bottom:1px solid #304058}th{position:sticky;top:0;background:#152239}td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}td:nth-child(2){max-width:480px;overflow-wrap:anywhere}.scroll{overflow:auto;max-height:650px}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{background:#17263b;padding:14px;border-radius:8px}.card strong{font-size:23px;display:block}.card span{color:#b0c0d4;font-size:12px}a{color:#84c9ff}</style></head><body>
<h1>Frozen WowRobo native flame profile</h1><p class="sub">Original accepted-v4 native source · checkpoint25 /966,743 ·0.25ms ×160substeps ·40ms action ·448 policy points. Two real workers, original PPO and SGS. Live training untouched.</p>
<div class="cards" id="cards"></div><h2>Flame graph</h2><select id="view"></select><button id="reset">Reset zoom</button><input id="search" placeholder="Search function or geometry"><div id="note"></div><div id="detail">Hover a frame for exact timing. Click to zoom.</div><div id="graph"></div>
<h2>Collision pair attribution</h2><p class="sub">Auxiliary native replay panel; each start is one40ms control interval. Calls and contact counts accumulate over160substeps. Budget is invoked initpoints ×iterations, not iterations actually executed. Supplement contacts are net additions; replacement of an existing contact does not increase this number. Tree nodes and leaf samples are actual existing traversal counters. Geometry and body names are authoritative; role labels are hints.</p><div class="scroll"><table id="pairs"></table></div>
<h2>Interpretation and reproducibility</h2><p class="sub">Sampled Python graphs measure wall time, including waits. A worker waiting for the learner and a learner waiting for workers overlap; their graph widths must not be added. The native graph is a separate instrumented timer hierarchy. Restricted kernel perf sampling prevented native call-stack unwinding. Pair replay instrumentation passed exact array parity; timing overhead is reported in the README. This two-world diagnostic preserves32sequential control rounds and PPO hyperparameters, but its small batch and local worker transport do not represent full-fleet throughput or network latency.</p><p><a href="README.md">Full report</a> · <a href="ppo-result.json">PPO record and worker telemetry</a> · <a href="instrumented-probe.json">Native pair counters</a> · <a href="parity.json">Parity checks</a> · <a href="learner.speedscope.json">Speedscope learner capture</a></p>
<script>const DATA=__PAYLOAD__;
const select=document.querySelector('#view');DATA.views.forEach((v,i)=>{let o=document.createElement('option');o.value=i;o.textContent=v.title;select.append(o)});let current,chosen,hover='';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function depth(n){return 1+Math.max(0,...n.children.map(depth))}function color(s){let h=0;for(let c of s)h=(h*31+c.charCodeAt(0))>>>0;return `hsl(${15+h%45},${70+h%20}%,${57+h%17}%)`}
function draw(){const w=Math.max(1000,document.querySelector('#graph').clientWidth),h=depth(current)*26+10,query=document.querySelector('#search').value.toLowerCase();let shapes=[];function walk(n,x,y,width){if(width<.4)return;const yy=h-(y+1)*26;shapes.push(`<g data-path="${esc(JSON.stringify(pathOf(n,current)||[]))}"><rect x="${x}" y="${yy}" width="${width}" height="24" fill="${query&&n.name.toLowerCase().includes(query)?'#f9f551':color(n.name)}"/><title>${esc(n.name)}\n${n.value.toFixed(6)} s · ${(100*n.value/chosen.tree.value).toFixed(2)}% of full view · ${(100*n.value/current.value).toFixed(2)}% of zoom</title>${width>35?`<text x="${x+4}" y="${yy+17}">${esc(n.name.slice(0,Math.floor(width/6.8)))}</text>`:''}</g>`);let xx=x;for(let c of n.children){let cw=width*c.value/n.value;walk(c,xx,y+1,cw);xx+=cw}}walk(current,0,0,w);document.querySelector('#graph').innerHTML=`<svg width="${w}" height="${h}">${shapes.join('')}</svg>`;document.querySelectorAll('svg g').forEach(g=>{g.onclick=()=>{let n=current;for(let i of JSON.parse(g.dataset.path))n=n.children[i];current=n;draw()};g.onmouseenter=()=>document.querySelector('#detail').textContent=g.querySelector('title').textContent})}
function pathOf(target,node){if(target===node)return [];for(let i=0;i<node.children.length;i++){const p=pathOf(target,node.children[i]);if(p)return [i,...p]}return null}
function choose(){chosen=DATA.views[Number(select.value)];current=chosen.tree;document.querySelector('#note').textContent=chosen.note;draw()};select.onchange=choose;document.querySelector('#reset').onclick=choose;document.querySelector('#search').oninput=draw;window.onresize=draw;
let r=DATA.result;document.querySelector('#cards').innerHTML=[[r.qualified?'PASS':'FAIL','Full PPO qualification'],[r.record.valid_transitions,'Valid transitions /64'],[r.elapsed_s.toFixed(2)+' s','Isolated full iteration'],[r.record.t_sim.toFixed(2)+' s','Collection'],[r.record.t_update.toFixed(3)+' s','PPO'],[DATA.parity.arrays,'Exactly matching replay arrays']].map(([v,k])=>`<div class="card"><strong>${v}</strong><span>${k}</span></div>`).join('');
const cols=[['start','Start'],['pair','Geometry pair'],['sdf_s','SDF s'],['supplement_s','Supplement s'],['calls','Calls'],['budget','Budget'],['tree_nodes','Tree nodes'],['leaf_samples','Leaf samples'],['aabb_rejected_nodes','AABB rejects'],['grid_rejected_nodes','Grid rejects'],['sdf_contacts','SDF contacts'],['returned_contacts','Returned contacts'],['returned_zero_calls','Zero-contact calls']];document.querySelector('#pairs').innerHTML='<thead><tr>'+cols.map(c=>`<th>${c[1]}</th>`).join('')+'</tr></thead><tbody>'+DATA.rows.map(r=>'<tr title="'+esc(r.bodies)+'">'+cols.map(([k])=>`<td>${k==='budget'?r.initpoints+'×'+r.iterations:typeof r[k]==='number'?(k.endsWith('_s')?r[k].toFixed(5):r[k].toLocaleString()):esc(r[k])}</td>`).join('')+'</tr>').join('')+'</tbody>';choose();
</script></body></html>'''
(HERE/'flamegraph.html').write_text(page.replace('__PAYLOAD__',payload))
print(HERE/'flamegraph.html')

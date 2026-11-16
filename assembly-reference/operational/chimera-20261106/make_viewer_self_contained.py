"""Render flame trees directly in the parent document for sandboxed previews."""
import base64
import json
from pathlib import Path
from render_flames import tree


def make_self_contained(out):
    path=out/'comparison.html'
    page=path.read_text()
    if '<iframe' not in page:
        (out/'comparison-standalone.html').write_text(page)
        return
    graphs={}
    for case in ['so101','assembly-nist']:
        for source in (out/case).glob('*.speedscope.json'):
            data=json.loads(source.read_text())
            graphs[case+'/'+source.with_suffix('.html').name]=dict(title=data['profiles'][0]['name'],root=tree(data))
    panel='''<section id="graph" aria-label="Interactive flame graph"><h3 id="flameTitle"></h3><div class="flameToolbar"><button id="flameReset">Reset zoom</button><input id="flameSearch" placeholder="Search function" aria-label="Search function"><span id="flameMeasure"></span></div><p id="flameTip" class="small">Hover over a frame for its full function name and time.</p><div id="flameCanvas"></div></section>'''
    page=page.replace('<iframe id="graph" title="Interactive flame graph"></iframe>',panel)
    page=page.replace('</style>','''#graph{border:1px solid #d6dfe7;background:white;border-radius:10px;padding:18px;min-height:420px}#flameCanvas{overflow-x:auto}#flameCanvas svg{display:block;width:100%;height:auto;min-width:760px}#flameCanvas svg text{pointer-events:none}#flameCanvas rect:hover{stroke:#17253a;stroke-width:1.5}.flameToolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}#flameSearch{padding:10px;border:1px solid #bcc9d6;border-radius:7px}#flameTip{min-height:40px;white-space:pre-wrap;overflow-wrap:anywhere}h3{margin-top:0}</style>''')
    js='''
const flameGraphs=FLAMES;let flameRoot=null,flameSelected=null,flameTimed=false;
function showFlame(key){const entry=flameGraphs[key];const host=document.querySelector('#flameCanvas');if(!entry){host.textContent='This profile is unavailable.';return}flameRoot=entry.root;flameSelected=flameRoot;flameTimed=entry.title.includes('phase timers');document.querySelector('#flameTitle').textContent=entry.title;document.querySelector('#flameSearch').value='';document.querySelector('#flameTip').textContent='Hover over a frame for its full function name and time.';paintFlame()}
function paintFlame(){if(!flameSelected)return;const host=document.querySelector('#flameCanvas');host.replaceChildren();const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('role','img');svg.setAttribute('aria-label','Zoomable flame graph');host.append(svg);let depth=0;const query=document.querySelector('#flameSearch').value.toLowerCase();function walk(node,x,w,d){if(w<.5)return;depth=Math.max(depth,d);const g=document.createElementNS(ns,'g'),r=document.createElementNS(ns,'rect'),title=document.createElementNS(ns,'title');let hash=0;for(const c of node.name)hash=(hash*31+c.charCodeAt(0))>>>0;const detail=`${node.name}\n${node.value.toFixed(4)} ${flameTimed?'timed':'sampled wall'} seconds · ${(100*node.value/flameRoot.value).toFixed(2)}% of full profile`;title.textContent=detail;g.append(title);for(const [k,v] of Object.entries({x:x,y:d*24,width:w,height:23,fill:query&&node.name.toLowerCase().includes(query)?'#aa70ec':`hsl(${12+hash%48} 80% 70%)`}))r.setAttribute(k,v);r.style.cursor='pointer';r.onclick=()=>{flameSelected=node;paintFlame()};r.onmousemove=()=>document.querySelector('#flameTip').textContent=detail;g.append(r);if(w>45){const t=document.createElementNS(ns,'text');t.setAttribute('x',x+4);t.setAttribute('y',d*24+16);t.setAttribute('font-size',12);t.textContent=node.name.replace(/^(py:|n:)/,'').split(' (')[0].slice(0,Math.floor(w/7)-2);g.append(t)}svg.append(g);let off=x;for(const child of node.children){const cw=w*child.value/node.value;walk(child,off,cw,d+1);off+=cw}}walk(flameSelected,0,1500,0);svg.setAttribute('viewBox',`0 0 1500 ${(depth+1)*24}`);svg.setAttribute('width',1500);svg.setAttribute('height',(depth+1)*24);document.querySelector('#flameMeasure').textContent=`${flameSelected.value.toFixed(2)} ${flameTimed?'timed':'sampled wall'} seconds shown`}
document.querySelector('#flameReset').onclick=()=>{flameSelected=flameRoot;paintFlame()};document.querySelector('#flameSearch').oninput=paintFlame;
'''.replace('FLAMES',json.dumps(graphs).replace('</','<\\/'))
    page=page.replace('<script>','<script>'+js,1)
    page=page.replace("document.querySelector('#graph').src=dataset+'/'+file;","showFlame(dataset+'/'+file);")
    for name in ['overview.png','worker-barriers.png']:
        url='data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()
        page=page.replace('src="'+name+'"','src="'+url+'"')
    assert '<iframe' not in page and 'fetch(' not in page
    path.write_text(page)
    (out/'comparison-standalone.html').write_text(page)

if __name__=='__main__':
    make_self_contained(Path(__file__).resolve().parents[2]/'outputs/ppo-profile-20261106')

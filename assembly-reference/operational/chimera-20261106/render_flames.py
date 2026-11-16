"""Standalone offline SVG and zoomable HTML flame graphs from sampled stacks."""
import hashlib
import html
import json
from pathlib import Path
import sys


def tree(data):
    root=dict(name='all timed phases' if any('phase timers' in p['name'] for p in data['profiles']) else 'all samples',value=0,children={})
    for profile in data['profiles']:
        for stack,weight in zip(profile['samples'],profile['weights']):
            node=root;node['value']+=weight
            for i in stack:
                name=data['shared']['frames'][i]['name']
                node=node['children'].setdefault(name,dict(name=name,value=0,children={}))
                node['value']+=weight
    def flatten(n):
        n['children']=[flatten(c) for c in sorted(n['children'].values(),key=lambda c:-c['value'])]
        return n
    return flatten(root)


def render(path):
    data=json.loads(path.read_text());root=tree(data)
    title=data['profiles'][0]['name']
    timed='phase timers' in title
    kind='timed' if timed else 'sampled'
    rects=[];depth=0;W=1500;H=23
    def walk(n,x,w,d):
        nonlocal depth
        if w<.5:return
        depth=max(depth,d)
        digest=hashlib.md5(n['name'].encode()).digest()
        color=f'hsl({12+digest[0]%48},80%,{60+digest[1]%16}%)'
        tip=html.escape(f"{n['name']} — {n['value']:.3f} {kind} seconds ({100*n['value']/root['value']:.2f}%)")
        rects.append(f'<g><title>{tip}</title><rect x="{x:.2f}" y="{55+d*H}" width="{w:.2f}" height="22" fill="{color}" stroke="white" stroke-width=".4"/>')
        if w>45:
            label=n['name'].split(' (')[0]
            if label.startswith(('py:','n:')):label=label.split(':')[1]
            label=html.escape(label[:max(1,int(w/7)-2)])
            rects.append(f'<text x="{x+4:.2f}" y="{71+d*H}" font-size="12">{label}</text>')
        rects.append('</g>')
        offset=x
        for c in n['children']:
            cw=w*c['value']/n['value'];walk(c,offset,cw,d+1);offset+=cw
    walk(root,0,W,0)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{95+(depth+1)*H}" style="font-family:Arial;background:#fff"><rect width="100%" height="100%" fill="#fff"/><text x="12" y="25" font-size="19" font-weight="bold">{html.escape(title)}</text><text x="12" y="45" font-size="13">Width = {kind} time; depth = call stack or timed phase hierarchy. Hover for details. {root["value"]:.2f} sampled seconds.</text>'+''.join(rects)+'</svg>'
    path.with_suffix('.svg').write_text(svg)
    document='''<!doctype html><meta charset="utf-8"><title>Flame graph</title><style>body{font:14px system-ui;margin:24px;background:#f7f8fb;color:#17253a}button,input{padding:8px;margin-right:8px}#graph{background:white;overflow:auto;border:1px solid #ddd}svg text{pointer-events:none}#tip{padding:12px;min-height:42px;white-space:pre-wrap}p{max-width:1000px}rect:hover{stroke:#111;stroke-width:2}</style><h1></h1><p>Click a frame to zoom; Reset restores the full profile. Width is sampled time, not elapsed position. Worker profiles sum main-thread wall samples across representative workers, including native call sites and waits; learner profiles sample wall-clock Python stacks, including waits. These measures must not be added together.</p><button id="reset">Reset</button><input id="search" placeholder="Search function"><span id="measure"></span><div id="tip"></div><div id="graph"></div><script>
const root=DATA,title=TITLE;document.querySelector('h1').textContent=title;let selected=root;const ns='http://www.w3.org/2000/svg';function draw(){const host=document.querySelector('#graph');host.replaceChildren();const svg=document.createElementNS(ns,'svg');svg.setAttribute('width',1500);host.append(svg);let depth=0;const query=document.querySelector('#search').value.toLowerCase();function walk(n,x,w,d){if(w<.5)return;depth=Math.max(depth,d);const g=document.createElementNS(ns,'g'),r=document.createElementNS(ns,'rect');r.setAttribute('x',x);r.setAttribute('y',d*24);r.setAttribute('width',w);r.setAttribute('height',23);let h=0;for(let c of n.name)h=(h*31+c.charCodeAt(0))>>>0;r.setAttribute('fill',query&&n.name.toLowerCase().includes(query)?'#aa70ec':`hsl(${12+h%48} 80% 70%)`);r.onclick=()=>{selected=n;draw()};r.onmousemove=()=>document.querySelector('#tip').textContent=`${n.name}\n${n.value.toFixed(4)} sampled seconds · ${(100*n.value/root.value).toFixed(2)}% of full profile`;g.append(r);if(w>45){let t=document.createElementNS(ns,'text');t.setAttribute('x',x+4);t.setAttribute('y',d*24+16);t.setAttribute('font-size',12);t.textContent=n.name.replace(/^(py:|n:)/,'').split(' (')[0].slice(0,Math.floor(w/7)-2);g.append(t)}svg.append(g);let off=x;for(let c of n.children){let cw=w*c.value/n.value;walk(c,off,cw,d+1);off+=cw}}walk(selected,0,1500,0);svg.setAttribute('height',(depth+1)*24);document.querySelector('#measure').textContent=`${selected.value.toFixed(2)} sampled seconds shown`};document.querySelector('#reset').onclick=()=>{selected=root;draw()};document.querySelector('#search').oninput=draw;draw();</script>'''
    if timed:document=document.replace('Click a frame to zoom;', 'MuJoCo native phase hierarchy from actual timers; this is not native stack sampling. Click a frame to zoom;').replace('sampled seconds','timed seconds').replace('Width is sampled time, not elapsed position.','Widths are timed, thread-summed phase durations, not elapsed position.')
    document=document.replace('DATA',json.dumps(root).replace('</','<\\/')).replace('TITLE',json.dumps(title))
    path.with_suffix('.html').write_text(document)


for name in sys.argv[1:]:render(Path(name))

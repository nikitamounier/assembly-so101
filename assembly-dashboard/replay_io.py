"""On-demand read-only replay export and vectorized visual pose reconstruction."""
import base64
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def model_fingerprint(path):
    """Identify the exact XML and mesh content independently of machine paths."""
    path = Path(path)
    model = ET.parse(path).getroot()
    if model.find('.//include') is not None:
        return None
    for element in model.iter():
        if 'file' not in element.attrib:
            continue
        asset = path.parent / element.attrib['file']
        element.set('file', hashlib.sha256(asset.read_bytes()).hexdigest())
    return hashlib.sha256(ET.tostring(model)).hexdigest()


def remote_clip(run, key, expected_id, include_model=False):
    root=Path(run)/'training-replays'
    manifest=json.loads((root/'manifest.json').read_text())
    index=json.loads((root/'index.json').read_text())
    versions=index['clips'][key]
    clip=versions.get('success') or versions.get('failure')
    assert expected_id is None or clip['id']==expected_id,'Replay updated; retry metadata observation'
    assert clip['complete_episode'] and clip['source']=='actual_training_episode'
    assert clip['identity']==manifest['identity'] and clip['identity']['run']==str(run)
    path=root/clip['file']
    assert path.parent.resolve()==root.resolve() and path.suffix=='.npz'
    raw=path.read_bytes();assert len(raw)<=16*1024*1024
    assert clip.get('data_sha256')==hashlib.sha256(raw).hexdigest(),'Replay slot updated; retry index'
    result = dict(metadata=clip,body_names=manifest['body_names'],data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
    if include_model:
        result['model_fingerprint'] = model_fingerprint(manifest['identity']['review_model'])
    return result


def remote_scene(run):
    import sys
    root=Path(run).parent
    sys.path.insert(0,str(root/'assembly-so101'))
    manifest=json.loads((Path(run)/'training-replays/manifest.json').read_text())
    return compile_scene(manifest['identity']['review_model'], manifest['identity'], manifest['body_names'])


def compile_scene(path, identity, body_names):
    import mujoco
    from assembly_policy.viewer.scene import visual_scene
    m=mujoco.MjModel.from_xml_path(str(path))
    names=[m.body(i).name for i in range(1,m.nbody)]
    assert names==body_names,'Replay body order differs from visual model'
    scene=visual_scene(m)
    layout=[]
    for geom in scene['geoms']:
        i=geom['id'];body=int(m.geom_bodyid[i]);name=m.body(body).name
        layout.append(dict(body=names.index(name) if body else -1,pos=m.geom_pos[i].tolist(),quat=m.geom_quat[i].tolist()))
    return dict(scene=scene,layout=layout,identity=identity)


def encode_scene(scene):
    """Pack mesh arrays losslessly in the types used by Three.js."""
    meshes = {
        name: dict(
            vertices=base64.b64encode(np.asarray(mesh['vertices'], dtype='<f4').tobytes()).decode(),
            faces=base64.b64encode(np.asarray(mesh['faces'], dtype='<u4').tobytes()).decode(),
        )
        for name, mesh in scene['meshes'].items()
    }
    return json.dumps({**scene, 'meshes': meshes, 'mesh_encoding': 'base64-le32'}, separators=(',', ':')).encode()


def reconstruct(payload, layout):
    raw=base64.b64decode(payload['data'],validate=True)
    assert hashlib.sha256(raw).hexdigest()==payload['sha256']
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        bodies=archive['frames'];times=archive['times']
    assert bodies.ndim==3 and bodies.shape[2]==7 and len(bodies)<=800
    assert bodies.shape[1]==len(payload['body_names']) and len(times)==len(bodies)
    assert np.isfinite(bodies).all() and np.isfinite(times).all()
    assert len(times)>=2 and times[0]==0 and np.all(np.diff(times)>0)
    indices=np.array([r['body'] for r in layout]);local_p=np.array([r['pos'] for r in layout]);local_q=np.array([r['quat'] for r in layout])
    p=bodies[:,np.maximum(indices,0),:3].copy();q=bodies[:,np.maximum(indices,0),3:].copy()
    p[:,indices<0]=0;q[:,indices<0]=[1,0,0,0]
    q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    v=np.broadcast_to(local_p,p.shape)
    cross=np.cross(q[...,1:],v)
    positions=p+v+2*(q[...,:1]*cross+np.cross(q[...,1:],cross))
    lq=np.broadcast_to(local_q,q.shape)
    scalar=q[...,:1]*lq[...,:1]-np.sum(q[...,1:]*lq[...,1:],axis=-1,keepdims=True)
    vector=q[...,:1]*lq[...,1:]+lq[...,:1]*q[...,1:]+np.cross(q[...,1:],lq[...,1:])
    frames=np.concatenate([positions,scalar,vector],axis=-1)
    return dict(**payload['metadata'],frames=np.round(frames,7).tolist(),times=times.tolist())

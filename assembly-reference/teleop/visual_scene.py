"""Read-only native visual geometry for local stereo rendering; no physics replica."""
import base64
import gzip
import json
import numpy as np
import mujoco


def packed(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode('ascii')


def scene_payload(model, owner_id):
    meshes, geoms = {}, []
    for g in range(model.ngeom):
        if model.geom_group[g] == 3 or model.geom_rgba[g,3] == 0:
            continue
        kind = mujoco.mjtGeom(int(model.geom_type[g])).name.removeprefix('mjGEOM_').lower()
        mesh = int(model.geom_dataid[g]) if kind in ('mesh','sdf') else None
        if mesh is not None and str(mesh) not in meshes:
            v, n = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            f, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            meshes[str(mesh)] = dict(position=packed(model.mesh_vert[v:v+n], '<f4'),
                                     index=packed(model.mesh_face[f:f+count], '<u4'))
        color = model.geom_rgba[g].copy()
        material = int(model.geom_matid[g])
        if material >= 0:
            color = model.mat_rgba[material].copy()
        item=dict(id=g,name=model.geom(g).name,body=int(model.geom_bodyid[g]),type=kind,mesh=mesh,
                          position=model.geom_pos[g].tolist(),quaternion=model.geom_quat[g].tolist(),
                          size=model.geom_size[g].tolist(),rgba=color.tolist())
        if material>=0 and (model.mat(material).name or '').startswith('hoto_'):
            item['surface']=dict(specular=float(model.mat_specular[material]),shininess=float(model.mat_shininess[material]))
        geoms.append(item)
    payload=dict(owner_id=owner_id,units='metres',body_count=model.nbody,meshes=meshes,geoms=geoms,
                 scope='Native visual meshes and measured body transforms; physics stays on host')
    return gzip.compress(json.dumps(payload,separators=(',',':')).encode(),compresslevel=3)

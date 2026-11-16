"""Experimental current-WowRobo Warp contacts; never imported by production.

Own process only: adapts the installed MJWarp SDF module in a private directory.
The GPU solver is float32; numerical/behavioral qualification is mandatory.
"""

from __future__ import annotations

import copy

# Warp needs explicit constructors for mutable loop variables.
# ruff: noqa: UP018, RUF046
import importlib.util
import os
import struct
import sys
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src import collision_driver, collision_sdf
from mujoco_warp._src.math import make_frame
from mujoco_warp._src.types import vec5, vec_pluginattr

# Native leaves use separate FP64 multiply/add and reciprocal multiplication.
# Contraction changes deepest-sample ties and signed-normal bins on flat faces.
wp.set_module_options({"enable_backward": False, "fuse_fp": False})
vec6d = wp.types.vector(length=6, dtype=wp.float64)
vec6i = wp.types.vector(length=6, dtype=wp.int32)


def _grid_native_source():
    default = Path(__file__).resolve().parents[3] / "assembly-so101/plugin/assemblenist_sdf.cc"
    source = Path(os.environ.get("WOWROBO_WARP_GRID_SOURCE", default)).read_text()
    body = source.split("double grid_eval(")[1].split("const char* kGridNames")[0]
    body = body[body.index("{") + 1 : body.rfind("}")]
    body = (
        body.replace("std::clamp", "wp::clamp")
        .replace("std::sqrt", "sqrt")
        .replace("std::floor", "floor")
    )
    body = body.replace("h2f(G->v[", "float(G->v[")
    body = body.replace(
        "return d + out;",
        "return wp::vec_t<4, double>(d+out, grad[0], grad[1], grad[2]);",
    )
    prefix = r"""
    auto bits = [&](int start) {
      uint64_t result = 0;
      for(int k=0;k<4;++k) result |= uint64_t(a[start+k]) << (16*k);
      return result;
    };
    auto number = [&](int start) { union {uint64_t u; double d;} value; value.u=bits(start); return value.d; };
    struct Grid {int nx,ny,nz;double ox,oy,oz,h;const float* v;};
    const Grid grid{int(a[1]),int(a[2]),int(a[3]),number(12),number(16),number(20),number(24),reinterpret_cast<const float*>(bits(8))};
    const Grid* G=&grid;
    double grad[3];
    """
    # The newer native fused evaluator adds an optional threshold argument.
    # This full-gradient port retains its original infinity default.
    if "gradient_threshold" in body:
        prefix += "const double gradient_threshold = INFINITY;\n"
    text = prefix + body
    for t in ("int", "float", "double"):
        text = text.replace(t + "(", "static_cast<" + t + ">(")
    return text


@wp.func_native(_grid_native_source())
def grid64(p: wp.vec3d, a: vec_pluginattr) -> wp.vec4d: ...


@wp.func
def user_sdf(p: wp.vec3, attr: vec_pluginattr, kind: int) -> float:
    return float(grid64(wp.vec3d(p), attr)[0])


@wp.func
def user_sdf_grad(p: wp.vec3, attr: vec_pluginattr, kind: int) -> wp.vec3:
    v = grid64(wp.vec3d(p), attr)
    return wp.vec3(float(v[1]), float(v[2]), float(v[3]))


@wp.kernel
def query_grid(
    points: wp.array[wp.vec3d],
    attr: wp.array[vec_pluginattr],
    ids: wp.array[int],
    out: wp.array[wp.vec4d],
):
    i = wp.tid()
    out[i] = grid64(points[i], attr[ids[i]])


@wp.kernel
def _samples(
    pairs: wp.array[wp.vec2i],
    worlds: wp.array[int],
    ncollision: wp.array[int],
    eligible: wp.array2d[int],
    visual: wp.array[int],
    adr: wp.array[wp.vec2i],
    points: wp.array[wp.vec3d],
    xpos: wp.array2d[wp.vec3],
    xmat: wp.array2d[wp.mat33],
    aabb: wp.array2d[wp.vec3d],
    plugins: wp.array[int],
    attrs: wp.array[vec_pluginattr],
    margins: wp.array2d[float],
    candidates: wp.array3d[vec6d],
    indices: wp.array3d[vec6i],
):
    cid, direction, lane = wp.tid()
    # _emit never consumes invalid capacity slots. _reduce has the same guard,
    # so their large sample scratch slabs can remain untouched between steps.
    if cid >= ncollision[0]:
        return
    pair = pairs[cid]
    if eligible[pair[0], pair[1]] == 0:
        return
    best = vec6d(wp.float64(1.0e30))
    bestid = vec6i(2147483647)
    if cid < ncollision[0]:
        geoms = pairs[cid]
        g1 = geoms[0]
        g2 = geoms[1]
        if eligible[g1, g2] != 0:
            w = worlds[cid]
            target = g1
            source = visual[g2]
            if direction == 1:
                target = g2
                source = visual[g1]
            start = adr[source][0]
            count = adr[source][1]
            rt = wp.mat33d(xmat[w, target])
            rs = wp.mat33d(xmat[w, source])
            pt = wp.vec3d(xpos[w, target])
            ps = wp.vec3d(xpos[w, source])
            center = aabb[target, 0]
            extent = aabb[target, 1]
            attr = attrs[plugins[target]]
            threshold = wp.float64(margins[g1, g2])
            i = lane
            while i < count:
                point = points[start + i]
                world = rs * point + ps
                local = wp.transpose(rt) * (world - pt)
                if (
                    wp.abs(local[0] - center[0]) <= extent[0] + wp.float64(0.0001)
                    and wp.abs(local[1] - center[1]) <= extent[1] + wp.float64(0.0001)
                    and wp.abs(local[2] - center[2]) <= extent[2] + wp.float64(0.0001)
                ):
                    value = grid64(local, attr)
                    dist = value[0]
                    grad = wp.vec3d(value[1], value[2], value[3])
                    length = wp.length(grad)
                    if (
                        wp.isfinite(dist)
                        and dist < threshold
                        and wp.isfinite(length)
                        and length >= wp.float64(1.0e-10)
                    ):
                        normal = rt * (grad * (wp.float64(1.0) / length))
                        axis = int(0)
                        for k in range(1, 3):
                            if wp.abs(normal[k]) > wp.abs(normal[axis]):
                                axis = k
                        b = 2 * axis + int(normal[axis] > wp.float64(0.0))
                        if dist < best[b] or (dist == best[b] and i < bestid[b]):
                            best[b] = dist
                            bestid[b] = i
                i += 128
    candidates[cid, direction, lane] = best
    indices[cid, direction, lane] = bestid


@wp.kernel
def _reduce(
    candidates: wp.array3d[vec6d],
    indices: wp.array3d[vec6i],
    winners: wp.array2d[int],
    distances: wp.array2d[wp.float64],
    pairs: wp.array[wp.vec2i],
    ncollision: wp.array[int],
    eligible: wp.array2d[int],
):
    cid, b = wp.tid()
    if cid >= ncollision[0]:
        return
    pair = pairs[cid]
    if eligible[pair[0], pair[1]] == 0:
        return
    direction = b // 6
    axis = b % 6
    best = wp.float64(1.0e30)
    idx = int(2147483647)
    for lane in range(128):
        dist = candidates[cid, direction, lane][axis]
        i = indices[cid, direction, lane][axis]
        if dist < best or (dist == best and i < idx):
            best = dist
            idx = i
    winners[cid, b] = idx
    distances[cid, b] = best


@wp.kernel
def _map_pairs(
    pairs: wp.array[wp.vec2i],
    worlds: wp.array[int],
    ncollision: wp.array[int],
    mapping: wp.array3d[int],
):
    cid = wp.tid()
    if cid < ncollision[0]:
        g = pairs[cid]
        w = worlds[cid]
        mapping[w, g[0], g[1]] = cid
        mapping[w, g[1], g[0]] = cid


@wp.kernel
def _map_contacts(
    geoms: wp.array[wp.vec2i],
    worlds: wp.array[int],
    nacon: wp.array[int],
    naconmax: int,
    mapping: wp.array3d[int],
    counts: wp.array[int],
    slots: wp.array2d[int],
    errors: wp.array[int],
):
    c = wp.tid()
    if c < wp.min(nacon[0], naconmax):
        g = geoms[c]
        w = worlds[c]
        if g[0] >= 0 and g[1] >= 0:
            cid = mapping[w, g[0], g[1]]
            if cid >= 0:
                i = wp.atomic_add(counts, cid, 1)
                if i < 50:
                    slots[cid, i] = c
                else:
                    wp.atomic_add(errors, 0, 1)


@wp.kernel
def _emit(
    pairs: wp.array[wp.vec2i],
    worlds: wp.array[int],
    ncollision: wp.array[int],
    pairids: wp.array[wp.vec2i],
    eligible: wp.array2d[int],
    visual: wp.array[int],
    adr: wp.array[wp.vec2i],
    points: wp.array[wp.vec3d],
    xpos: wp.array2d[wp.vec3],
    xmat: wp.array2d[wp.mat33],
    plugins: wp.array[int],
    attrs: wp.array[vec_pluginattr],
    winners: wp.array2d[int],
    distances: wp.array2d[wp.float64],
    counts: wp.array[int],
    slots: wp.array2d[int],
    # Pre-mixed native material laws for every pair (no physics tuning).
    margin: wp.array2d[float],
    gap: wp.array2d[float],
    dim: wp.array2d[int],
    friction: wp.array2d[vec5],
    solref: wp.array2d[wp.vec2],
    solreffriction: wp.array2d[wp.vec2],
    solimp: wp.array2d[vec5],
    nacon: wp.array[int],
    naconmax: int,
    cdist: wp.array[float],
    cpos: wp.array[wp.vec3],
    cframe: wp.array[wp.mat33],
    cmargin: wp.array[float],
    cfriction: wp.array[vec5],
    csolref: wp.array[wp.vec2],
    csolreffriction: wp.array[wp.vec2],
    csolimp: wp.array[vec5],
    cdim: wp.array[int],
    cgeom: wp.array[wp.vec2i],
    cefc: wp.array2d[int],
    cworld: wp.array[int],
    ctype: wp.array[int],
    ccollision: wp.array[int],
    cadhesion: wp.array[float],
    cflex: wp.array[wp.vec2i],
    celem: wp.array[wp.vec2i],
    cvert: wp.array[wp.vec2i],
    csurface: wp.array[wp.vec3],
    supplement_count: wp.array[int],
):
    cid = wp.tid()
    if cid >= ncollision[0]:
        return
    gs = pairs[cid]
    g1 = gs[0]
    g2 = gs[1]
    w = worlds[cid]
    if eligible[g1, g2] == 0:
        return
    count = counts[cid]
    for b in range(12):
        distance = distances[cid, b]
        if distance >= wp.float64(margin[g1, g2]):
            continue
        direction = b // 6
        target = g1
        source = visual[g2]
        sign = wp.float64(1.0)
        if direction == 1:
            target = g2
            source = visual[g1]
            sign = wp.float64(-1.0)
        i = winners[cid, b]
        point = points[adr[source][0] + i]
        world = wp.mat33d(xmat[w, source]) * point + wp.vec3d(xpos[w, source])
        rt = wp.mat33d(xmat[w, target])
        local = wp.transpose(rt) * (world - wp.vec3d(xpos[w, target]))
        v = grid64(local, attrs[plugins[target]])
        grad = wp.vec3d(v[1], v[2], v[3])
        normal = rt * (grad * (wp.float64(1.0) / wp.length(grad)))
        pos = world - wp.float64(0.5) * distance * normal
        dest = int(-1)
        if count < 50:
            dest = wp.atomic_add(nacon, 0, 1)
            if dest < naconmax:
                slots[cid, count] = dest
            else:
                # Leave nacon over capacity for the caller's latched guard.
                # Never revisit an uninitialized slot in the replacement loop.
                return
            count += 1
        else:
            shallow = int(0)
            for j in range(1, 50):
                if cdist[slots[cid, j]] > cdist[slots[cid, shallow]]:
                    shallow = j
            if float(distance) < cdist[slots[cid, shallow]]:
                dest = slots[cid, shallow]
        if dest >= 0 and dest < naconmax:
            cdist[dest] = float(distance)
            cpos[dest] = wp.vec3(pos)
            cframe[dest] = make_frame(wp.vec3(sign * normal))
            cmargin[dest] = margin[g1, g2] - gap[g1, g2]
            cfriction[dest] = friction[g1, g2]
            csolref[dest] = solref[g1, g2]
            csolreffriction[dest] = solreffriction[g1, g2]
            csolimp[dest] = solimp[g1, g2]
            cdim[dest] = dim[g1, g2]
            cgeom[dest] = gs
            cworld[dest] = w
            # Match stock ContactType.CONSTRAINT / SENSOR bits.
            typ = int(0)
            if pairids[cid][0] >= -1:
                typ = 1
            if pairids[cid][1] >= 0:
                typ = typ | 2
            ctype[dest] = typ
            ccollision[dest] = 40 + b
            cadhesion[dest] = 0.0
            if dest < cflex.shape[0]:
                cflex[dest] = wp.vec2i(-1)
            if dest < celem.shape[0]:
                celem[dest] = wp.vec2i(-1)
            if dest < cvert.shape[0]:
                cvert[dest] = wp.vec2i(-1)
            if dest < csurface.shape[0]:
                csurface[dest] = wp.vec3(0.0)
            for k in range(cefc.shape[1]):
                cefc[dest, k] = -1
            wp.atomic_add(supplement_count, w, 1)
    counts[cid] = count


@wp.kernel
def _friction(
    geoms: wp.array[wp.vec2i],
    worlds: wp.array[int],
    nacon: wp.array[int],
    naconmax: int,
    roles: wp.array[int],
    draws: wp.array2d[float],
    friction: wp.array[vec5],
):
    c = wp.tid()
    if c >= wp.min(nacon[0], naconmax):
        return
    g = geoms[c]
    if g[0] < 0 or g[1] < 0:
        return
    a = roles[g[0]]
    b = roles[g[1]]
    k = int(-1)
    if (a == 1 and b == 2) or (a == 2 and b == 1):
        k = 0
    elif (a == 2 and b == 3) or (a == 3 and b == 2):
        k = 1
    elif a == 2 and b == 2:
        k = 2
    if k >= 0:
        f = friction[c]
        scale = draws[worlds[c], k]
        f[0] *= scale
        f[1] *= scale
        friction[c] = f


@wp.kernel
def monitor_contacts(
    geoms: wp.array[wp.vec2i],
    worlds: wp.array[int],
    nacon: wp.array[int],
    naconmax: int,
    distance: wp.array[float],
    peak: wp.array[float],
):
    c = wp.tid()
    if c < wp.min(nacon[0], naconmax):
        wp.atomic_min(peak, worlds[c], distance[c])


def _numeric(m, name, fallback):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    return (
        np.asarray(
            m.numeric_data[m.numeric_adr[i] : m.numeric_adr[i] + m.numeric_size[i]]
        )
        if i >= 0
        else np.array(fallback)
    )


def _pair_tables(m):
    """Native routing and material law tables, including geom-priority mixing."""
    n = m.ngeom
    visual = np.full(n, -1, np.int32)
    eligible = np.zeros((n, n), np.int32)
    budget = np.zeros((n, n, 2), np.int32)
    margins = np.zeros((n, n), np.float32)
    gaps = margins.copy()
    dims = np.zeros((n, n), np.int32)
    fr = np.zeros((n, n, 5), np.float32)
    sr = np.zeros((n, n, 2), np.float32)
    srf = sr.copy()
    si = fr.copy()
    explicit = {
        frozenset(map(int, (m.pair_geom1[i], m.pair_geom2[i]))): i
        for i in range(m.npair)
    }
    flag = bool(_numeric(m, "wowrobo_surface_contact_guard", [0])[0])
    for g in range(n):
        name = m.geom(g).name or ""
        if name.endswith("__contact"):
            visual[g] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name[:-9])

    def finger(g):
        name = m.geom(g).name or ""
        return (
            "wrist_roll_follower_so101_v1_collision_sdf" in name
            or "moving_jaw_so101_v1_collision_sdf" in name
        )

    def free(g):
        b = m.body_rootid[m.geom_bodyid[g]]
        return (
            b > 0
            and m.body_jntnum[b] > 0
            and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE
        )

    for a in range(n):
        for b in range(n):
            generic = flag and visual[a] >= 0 and visual[b] >= 0
            eligible[a, b] = int(generic)
            key = "so101_sdf_search"
            if generic:
                key = "so101_assembly_search"
            elif (finger(a) and free(b)) or (finger(b) and free(a)):
                key = "so101_grasp_search"
            elif free(a) and free(b):
                key = "so101_assembly_search"
            budget[a, b] = _numeric(
                m, key, [m.opt.sdf_initpoints, m.opt.sdf_iterations]
            )[:2]
            pair = explicit.get(frozenset((a, b)))
            if pair is not None:
                margins[a, b] = m.pair_margin[pair]
                gaps[a, b] = m.pair_gap[pair]
                dims[a, b] = m.pair_dim[pair]
                fr[a, b] = m.pair_friction[pair]
                sr[a, b] = m.pair_solref[pair]
                srf[a, b] = m.pair_solreffriction[pair]
                si[a, b] = m.pair_solimp[pair]
            else:
                margins[a, b] = m.geom_margin[a] + m.geom_margin[b]
                gaps[a, b] = m.geom_gap[a] + m.geom_gap[b]
                pa, pb = int(m.geom_priority[a]), int(m.geom_priority[b])
                if pa != pb:
                    g = a if pa > pb else b
                    dims[a, b] = m.geom_condim[g]
                    f = m.geom_friction[g]
                    sr[a, b] = m.geom_solref[g]
                    si[a, b] = m.geom_solimp[g]
                else:
                    dims[a, b] = max(m.geom_condim[a], m.geom_condim[b])
                    f = np.maximum(m.geom_friction[a], m.geom_friction[b])
                    mix = m.geom_solmix[a] + m.geom_solmix[b]
                    weight = m.geom_solmix[a] / mix if mix > 1.0e-15 else 0.5
                    sr[a, b] = (
                        weight * m.geom_solref[a] + (1 - weight) * m.geom_solref[b]
                        if np.all(m.geom_solref[[a, b], 0] > 0)
                        else np.minimum(m.geom_solref[a], m.geom_solref[b])
                    )
                    si[a, b] = (
                        weight * m.geom_solimp[a] + (1 - weight) * m.geom_solimp[b]
                    )
                fr[a, b] = [f[0], f[0], f[1], f[2], f[2]]
    return visual, eligible, budget, margins, gaps, dims, fr, sr, srf, si


class WowRoboWarp:
    def __init__(
        self,
        m,
        reference_data,
        grid_directory,
        scratch,
        nworld=1,
        nconmax=512,
        njmax=2048,
        jacobian="native",
    ):
        self.native = m
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        if jacobian not in {"native", "dense"}:
            raise ValueError("jacobian must be native or dense")
        gpu_source, gpu_initial = m, reference_data
        if jacobian == "dense":
            # Storage representation only: preserve every mechanical parameter,
            # solver limit, tolerance, contact and integration state.
            gpu_source = copy.copy(m)
            gpu_source.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
            gpu_initial = mujoco.MjData(gpu_source)
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(m, spec), dtype=np.float64)
            mujoco.mj_getState(m, reference_data, state, spec)
            mujoco.mj_setState(gpu_source, gpu_initial, state, spec)
        self.gpu_source = gpu_source
        self.model = mjw.put_model(gpu_source)
        self.model.opt.tolerance.fill_(float(m.opt.tolerance))
        self.model.opt.contact_sensor_maxmatch = nconmax
        self.draws = wp.ones((nworld, 3), dtype=float)
        self.grids = {}
        attrs = self.model.plugin_attr.numpy()
        geom_indices = self.model.geom_plugin_index.numpy()
        for geom in np.flatnonzero(m.geom_plugin >= 0):
            idx = int(geom_indices[geom])
            grid_id = int(attrs[idx, 0])
            path = Path(grid_directory) / f"grid_{grid_id}.bin"
            with path.open("rb") as f:
                shape = np.fromfile(f, np.int32, 3)
                header = np.fromfile(f, np.float64, 4)
                values = np.fromfile(f, np.float16).astype(np.float32)
            if len(values) != int(np.prod(shape)):
                raise ValueError("Malformed SDF grid")
            if grid_id not in self.grids:
                self.grids[grid_id] = (shape, header, wp.array(values, dtype=float))
            shape, header, array = self.grids[grid_id]
            attrs[idx, 1:4] = shape
            attrs[idx, 8:12] = [(array.ptr >> (16 * k)) & 65535 for k in range(4)]
            for k, v in enumerate(header):
                bits = struct.unpack("<Q", struct.pack("<d", v))[0]
                attrs[idx, 12 + 4 * k : 16 + 4 * k] = [
                    (bits >> (16 * j)) & 65535 for j in range(4)
                ]
        self.model.plugin_attr = wp.array(attrs, dtype=vec_pluginattr)
        self.model.plugin = wp.zeros(len(attrs), dtype=int)
        (visual, eligible, budget, margin, gap, dim, fr, sr, srf, si) = _pair_tables(m)
        self.table_audit = {
            "eligible_pairs": int(np.count_nonzero(np.triu(eligible, 1))),
            "budgets": np.unique(budget.reshape(-1, 2), axis=0).tolist(),
        }
        self.visual = wp.array(visual, dtype=int)
        self.eligible = wp.array(eligible, dtype=int)
        self.budget = wp.array(budget, dtype=wp.vec2i)
        self.margin = wp.array(margin, dtype=float)
        self.gap = wp.array(gap, dtype=float)
        self.dim = wp.array(dim, dtype=int)
        self.fr = wp.array(fr, dtype=vec5)
        self.sr = wp.array(sr, dtype=wp.vec2)
        self.srf = wp.array(srf, dtype=wp.vec2)
        self.si = wp.array(si, dtype=vec5)
        # Keep native sample order: mesh vertices followed by triangle centroids.
        points = []
        adr = np.zeros((m.ngeom, 2), np.int32)
        for g in sorted({int(x) for x in visual if x >= 0}):
            mesh = int(m.geom_dataid[g])
            v0 = int(m.mesh_vertadr[mesh])
            vn = int(m.mesh_vertnum[mesh])
            f0 = int(m.mesh_faceadr[mesh])
            fn = int(m.mesh_facenum[mesh])
            verts_raw = m.mesh_vert[v0 : v0 + vn]
            verts = verts_raw.astype(np.float64)
            faces = m.mesh_face[f0 : f0 + fn]
            centers = np.zeros((fn, 3), np.float64)
            for j in range(3):
                # Native divides each float32 vertex before double accumulation.
                centers += (verts_raw[faces[:, j]] / np.float32(3.0)).astype(np.float64)
            adr[g] = [sum(len(x) for x in points), vn + fn]
            points.extend([verts, centers])
        self.points = wp.array(np.concatenate(points), dtype=wp.vec3d)
        self.adr = wp.array(adr, dtype=wp.vec2i)
        self.aabb = wp.array(
            np.array(m.geom_aabb).reshape(m.ngeom, 2, 3), dtype=wp.vec3d
        )
        self._install_search(budget)
        self.data = mjw.put_data(
            gpu_source, gpu_initial, nworld=nworld, nconmax=nconmax, njmax=njmax
        )
        cap = self.data.naconmax
        self.candidates = wp.empty((cap, 2, 128), dtype=vec6d)
        self.indices = wp.empty((cap, 2, 128), dtype=vec6i)
        self.winners = wp.empty((cap, 12), dtype=int)
        self.distances = wp.empty((cap, 12), dtype=wp.float64)
        self.mapping = wp.full((nworld, m.ngeom, m.ngeom), -1, dtype=int)
        self.counts = wp.zeros(cap, dtype=int)
        self.slots = wp.zeros((cap, 50), dtype=int)
        self.errors = wp.zeros(1, dtype=int)
        self.supplement_count = wp.zeros(nworld, dtype=int)
        # Use the unchanged material-role inference from the isolated snapshot.
        spec = importlib.util.spec_from_file_location(
            "_warp_roles", os.environ.get("WOWROBO_WARP_FRICTION_SOURCE", str(Path(__file__).with_name("wowrobo_friction.py")))
        )
        rolesmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rolesmod)
        roles = rolesmod.infer_geom_roles(m)
        codes = {"finger": 1, "part": 2, "table": 3}
        self.roles = wp.array(
            np.array([codes.get(roles.get(i), 0) for i in range(m.ngeom)], np.int32),
            dtype=int,
        )
        self.model.callback.contactfilter = self._contacts

    def _install_search(self, budget):
        self._original_search = collision_driver.sdf_narrowphase
        source = Path(collision_sdf.__file__).read_text()
        source = source.replace(
            '"enable_backward": False, "default_grid_stride": False',
            '"enable_backward": False, "default_grid_stride": False, "fuse_fp": False',
        )
        old = "  sdf_initpoints: int,\n  sdf_iterations: int,\n  linesearch_alpha0: float,"
        assert source.count(old) == 1
        source = source.replace(old, "  pair_budget: wp.array2d[wp.vec2i],\n" + old)
        old = "  geoms = collision_pair_in[contact_tid]\n  g2 = geoms[1]"
        assert source.count(old) == 1
        source = source.replace(
            old,
            "  geoms = collision_pair_in[contact_tid]\n  budget_values = pair_budget[geoms[0], geoms[1]]\n  if i >= budget_values[0]:\n    return\n  g2 = geoms[1]",
        )
        old = "    sdf_iterations,\n    linesearch_alpha0,"
        assert source.count(old) == 1
        source = source.replace(old, "    budget_values[1],\n    linesearch_alpha0,")
        source = source.replace(
            "      m.opt.sdf_initpoints,\n      m.opt.sdf_iterations,",
            "      PAIR_BUDGET,\n      m.opt.sdf_initpoints,\n      m.opt.sdf_iterations,",
        )
        path = self.scratch / "private_collision_sdf.py"
        path.write_text(source)
        name = "_wowrobo_warp_collision_sdf"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mod.PAIR_BUDGET = self.budget
        mod.user_sdf = user_sdf
        mod.user_sdf_grad = user_sdf_grad
        collision_driver.sdf_narrowphase = mod.sdf_narrowphase
        self.model.opt.sdf_initpoints = int(budget[:, :, 0].max())
        self.model.opt.sdf_iterations = int(budget[:, :, 1].max())
        self.search_module = mod

    def _contacts(self, m, d):
        ctx = self.context
        cap = d.naconmax
        self.mapping.fill_(-1)
        self.counts.zero_()
        self.errors.zero_()
        self.supplement_count.zero_()
        wp.launch(
            _samples,
            dim=(cap, 2, 128),
            inputs=[
                ctx.collision_pair,
                ctx.collision_worldid,
                d.ncollision,
                self.eligible,
                self.visual,
                self.adr,
                self.points,
                d.geom_xpos,
                d.geom_xmat,
                self.aabb,
                m.geom_plugin_index,
                m.plugin_attr,
                self.margin,
                self.candidates,
                self.indices,
            ],
        )
        wp.launch(
            _reduce,
            dim=(cap, 12),
            inputs=[self.candidates, self.indices, self.winners, self.distances,
                    ctx.collision_pair, d.ncollision, self.eligible],
        )
        wp.launch(
            _map_pairs,
            dim=cap,
            inputs=[
                ctx.collision_pair,
                ctx.collision_worldid,
                d.ncollision,
                self.mapping,
            ],
        )
        wp.launch(
            _map_contacts,
            dim=cap,
            inputs=[
                d.contact.geom,
                d.contact.worldid,
                d.nacon,
                cap,
                self.mapping,
                self.counts,
                self.slots,
                self.errors,
            ],
        )
        c = d.contact
        wp.launch(
            _emit,
            dim=cap,
            inputs=[
                ctx.collision_pair,
                ctx.collision_worldid,
                d.ncollision,
                ctx.collision_pairid,
                self.eligible,
                self.visual,
                self.adr,
                self.points,
                d.geom_xpos,
                d.geom_xmat,
                m.geom_plugin_index,
                m.plugin_attr,
                self.winners,
                self.distances,
                self.counts,
                self.slots,
                self.margin,
                self.gap,
                self.dim,
                self.fr,
                self.sr,
                self.srf,
                self.si,
                d.nacon,
                cap,
                c.dist,
                c.pos,
                c.frame,
                c.includemargin,
                c.friction,
                c.solref,
                c.solreffriction,
                c.solimp,
                c.dim,
                c.geom,
                c.efc_address,
                c.worldid,
                c.type,
                c.geomcollisionid,
                c.adhesion,
                c.flex,
                c.elem,
                c.vert,
                c.surfacevel,
                self.supplement_count,
            ],
        )
        wp.launch(
            _friction,
            dim=cap,
            inputs=[
                c.geom,
                c.worldid,
                d.nacon,
                cap,
                self.roles,
                self.draws,
                c.friction,
            ],
        )

    def install_context_capture(self):
        # Keep each collision context alive for CUDA graph replay and callback use.
        original = collision_driver.create_collision_context
        self._original_context = original

        def create(n):
            self.context = original(n)
            return self.context

        collision_driver.create_collision_context = create
        self._context_factory = create

    def close(self):
        """Release process-global hooks after all outstanding GPU work ends."""
        wp.synchronize()
        if collision_driver.sdf_narrowphase is self.search_module.sdf_narrowphase:
            collision_driver.sdf_narrowphase = self._original_search
        if collision_driver.create_collision_context is getattr(self, "_context_factory", None):
            collision_driver.create_collision_context = self._original_context
        self.model.callback.contactfilter = None

    def check(self):
        if int(self.data.nacon.numpy()[0]) > self.data.naconmax:
            raise RuntimeError("GPU contact overflow")
        if int(self.data.ncollision.numpy()[0]) > self.data.naconmax:
            raise RuntimeError("GPU broadphase overflow")
        if np.any(self.data.nefc.numpy() > self.data.njmax):
            raise RuntimeError("GPU constraint overflow")
        if np.any(self.data.overflow.numpy()):
            raise RuntimeError(
                "GPU overflow flags: " + str(self.data.overflow.numpy().tolist())
            )
        if int(self.errors.numpy()[0]):
            raise RuntimeError("Ordinary manifold exceeds50contacts/pair")
        if not np.isfinite(self.data.qpos.numpy()).all():
            raise RuntimeError("Nonfinite GPU state")

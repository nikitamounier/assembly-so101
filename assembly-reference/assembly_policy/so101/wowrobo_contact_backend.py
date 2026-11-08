"""Isolated WowRobo source-mesh SDF profile with accelerated native supplements.

Keeps the original pose-bank degrees of freedom. This is a fixture-probe backend,
not training admission or a qualification of the generated distance fields.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "assembly-so101"
OUTPUT = ROOT / "outputs/linear-insertion-updated-20261107"
DEFAULT_NATIVE = OUTPUT / "native"
_LIBRARIES = {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_native_unlocked(build_dir=DEFAULT_NATIVE):
    """Copy accelerated sources and patch pair routing, never the runtime plugin."""
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "assemblenist_sdf.cc",
        "so101_grid_bounds.h",
        "so101_surface_tree.h",
        "so101_contact_search.c",
        "build.sh",
        "build_contact_search.sh",
    ]
    originals = {name: (SOURCE / "plugin" / name).read_bytes() for name in names}
    text = originals["so101_contact_search.c"].decode()
    optimization = (Path(__file__).parent / "wowrobo_contact_cache.h").read_text()
    text = optimization + "\n" + text
    anchor = '  int forward = !strcmp(a, "base_contact")'
    start = text.index(anchor)
    stop = text.index("  mjtNum threshold = margin;", start)
    original = text[start:stop]
    generic = """  // WowRobo routing only: sampling/binning/culling below is unchanged.
  int forward = 0, reverse = 0;
  int first_visual = -1, second_visual = -1;
  const char* suffix = "__contact";
  size_t na = strlen(a), nb = strlen(b), ns = strlen(suffix);
  int generic_flag = mj_name2id(m, mjOBJ_NUMERIC, "wowrobo_surface_contact_guard");
  if (generic_flag >= 0 && m->numeric_data[m->numeric_adr[generic_flag]] != 0 &&
      na > ns && nb > ns && !strcmp(a+na-ns,suffix) && !strcmp(b+nb-ns,suffix)) {
    std::string av(a,na-ns), bv(b,nb-ns);
    first_visual = mj_name2id(m,mjOBJ_GEOM,av.c_str());
    second_visual = mj_name2id(m,mjOBJ_GEOM,bv.c_str());
    forward = 1;
  } else {
"""
    # Preserve the old routes too; shadow only local route temporaries.
    old = original.replace("  int forward =", "  forward =", 1).replace(
        "  int reverse =", "  reverse =", 1
    )
    old = old.replace("  int first_visual =", "  first_visual =", 1).replace(
        "  int second_visual =", "  second_visual =", 1
    )
    text = (
        text[:start]
        + generic
        + old
        + "  }\n  if (first_visual < 0 || second_visual < 0) return count;\n"
        + text[stop:]
    )
    dispatch = '  if (a && b && ((!strcmp(a, "base_contact")'
    at = text.index(dispatch, text.index("static int contact_search"))
    text = (
        text[:at]
        + """  int wow_helper_role = 0;
  int wow = mj_name2id(m, mjOBJ_NUMERIC, "wowrobo_surface_contact_guard");
  if (wow >= 0 && a && b && strstr(a,"__contact") && strstr(b,"__contact")) {
    int assembly = mj_name2id(m,mjOBJ_NUMERIC,"so101_assembly_search");
    if (assembly >= 0) id=assembly;
    // Preserve explicit mating/passive-stop budgets. Helpers have separate
    // grasp and hardware budgets; original target-target mating stays 40x40.
    int helper_a=strstr(a,"physical_helper_")!=nullptr;
    int helper_b=strstr(b,"physical_helper_")!=nullptr;
    int role=mj_name2id(m,mjOBJ_NUMERIC,
        helper_a && helper_b ? "wowrobo_hardware_search" : "wowrobo_grasp_search");
    bool explicit_pair=false;
    for(int pair=0;pair<m->npair;++pair)
      explicit_pair|=(m->pair_geom1[pair]==g1 && m->pair_geom2[pair]==g2) ||
                     (m->pair_geom1[pair]==g2 && m->pair_geom2[pair]==g1);
    if ((helper_a || helper_b) && !explicit_pair) {
      if(role>=0) id=role;
      else wow_helper_role=helper_a && helper_b ? 2 : 1;
    }
  } else
"""
        + text[at:]
    )
    # Exact-pose memoization applies only to our immutable grid-SDF supplement.
    # Ordinary MuJoCo collision search still runs on every call.
    insertion = "  for (int direction = 0; direction < 2; direction++) {"
    text = text.replace(insertion, """  WowSupplementMemo* memo = wow_supplement_memo(m,d,g1,g2,threshold);
  if (memo->matches && surface_mode != 2) {
    memcpy(bins,memo->bins,sizeof(bins));
  } else {
""" + insertion, 1)
    merge = "  for (int i = 0; i < 12; i++) {\n    if (bins[i].dist >= threshold) continue;"
    text = text.replace(merge, """  memcpy(memo->bins,bins,sizeof(bins));
  memo->valid=true;
  }
""" + merge, 1)
    budget_anchor = "  view.opt.sdf_iterations = (int)values[1];"
    text = text.replace(budget_anchor, budget_anchor + """
  if (wow_helper_role) {
    view.opt.sdf_initpoints = 40;
    view.opt.sdf_iterations = 40;
  }
""", 1)
    collider_call = "  int count = collider(&view, d, con, g1, g2, margin);"
    assert text.count(collider_call)==1
    text = text.replace(collider_call, "  int count = surface_mode!=2 && margin<=0 && wow_sdf_pair_clear(m,d,g1,g2) ? 0 : collider(&view, d, con, g1, g2, margin);")
    text += r"""
extern "C" double wowrobo_sdf_distance(const mjModel* m, const mjData* d, int geom, const mjtNum world[3]) {
  mjtNum local[3], delta[3];
  mju_sub3(delta,world,d->geom_xpos+3*geom);
  mju_mulMatTVec3(local,d->geom_xmat+9*geom,delta);
  int instance=m->geom_plugin[geom];
  if (instance<0) return NAN;
  const mjpPlugin* p=mjp_getPluginAtSlot(m->plugin[instance]);
  return p && p->sdf_distance ? p->sdf_distance(local,d,instance) : NAN;
}
"""
    text = "#include <string>\n" + text
    for name, data in originals.items():
        target = build_dir / name
        desired = text.encode() if name == "so101_contact_search.c" else data
        if not target.exists() or target.read_bytes() != desired:
            target.write_bytes(desired)
    identity = {
        name: hashlib.sha256(data).hexdigest() for name, data in originals.items()
    }
    desired_id = {
        "source": str(SOURCE / "plugin"),
        "source_hashes": identity,
        "patched_contact_sha256": sha256(build_dir / "so101_contact_search.c"),
        "optimization_header_sha256": hashlib.sha256(optimization.encode()).hexdigest(),
        "routing": "numeric wowrobo_surface_contact_guard + original visual names / __contact suffix",
        "unchanged_algorithms": [
            "grid_eval",
            "so101_grid_bounds.h",
            "so101_surface_tree.h",
            "sample_surface",
            "supplement bins and original sample iteration order",
        ],
    }
    identity_file = build_dir / "build-receipt.json"
    cached = json.loads(identity_file.read_text()) if identity_file.exists() else {}
    libraries = [
        build_dir / "build/libassemblenist_sdf.so",
        build_dir / "build/libso101_contact_search.so",
    ]
    valid_libraries = all(
        x.exists() and cached.get("libraries", {}).get(str(x)) == sha256(x)
        for x in libraries
    )
    if cached.get("identity") != desired_id or not valid_libraries:
        env = dict(os.environ, PYTHON=sys.executable)
        for script in ["build.sh", "build_contact_search.sh"]:
            subprocess.run(
                ["bash", str(build_dir / script)], cwd=build_dir, env=env, check=True
            )
        identity_file.write_text(
            json.dumps(
                {
                    "identity": desired_id,
                    "libraries": {str(x): sha256(x) for x in libraries},
                    "mujoco_version": mujoco.__version__,
                },
                indent=2,
            )
            + "\n"
        )
    return build_dir


def build_native(build_dir=DEFAULT_NATIVE):
    """Serialize builds and verify both source and compiled-library identities."""
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    with (build_dir / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_native_unlocked(build_dir)


def load_backend(grid_manifest, build_dir=DEFAULT_NATIVE):
    """Must precede model compilation in each isolated Python process."""
    manifest_path = Path(grid_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    grid_dir = Path(
        manifest.get("grid_dir", manifest.get("grid_directory", manifest_path.parent))
    )
    if not grid_dir.is_absolute():
        grid_dir = manifest_path.parent / grid_dir
    for item in manifest["meshes"].values():
        grid_id = item.get("grid_id", item.get("id"))
        if not (grid_dir / f"grid_{grid_id}.bin").exists():
            raise FileNotFoundError(grid_dir / f"grid_{grid_id}.bin")
    key = (str(grid_dir), str(Path(build_dir).resolve()))
    if _LIBRARIES and key not in _LIBRARIES:
        raise RuntimeError(
            "Grid libraries are immutable; use a fresh process for another profile"
        )
    if key not in _LIBRARIES:
        native = build_native(build_dir)
        os.environ["ASSEMBLENIST_SDF_DIR"] = str(grid_dir)
        # The same registration names are intentional: run in a fresh process,
        # with no historical SDF library loaded.
        mujoco.mj_loadPluginLibrary(str(native / "build/libassemblenist_sdf.so"))
        contact = ctypes.CDLL(
            str(native / "build/libso101_contact_search.so"), mode=ctypes.RTLD_GLOBAL
        )
        if contact.so101_install_contact_search() != 4:
            raise RuntimeError("Wrong native adapter ABI")
        _LIBRARIES[key] = contact
    return {
        "native_dir": str(Path(build_dir).resolve()),
        "grid_dir": str(grid_dir),
        "grid_manifest_sha256": sha256(manifest_path),
        "build": json.loads((Path(build_dir) / "build-receipt.json").read_text()),
    }


def surface_library():
    if not _LIBRARIES:
        raise RuntimeError("load_backend must be called first")
    return next(iter(_LIBRARIES.values()))


def set_surface_mode(mode):
    if mode not in (0, 1, 2):
        raise ValueError("surface mode must be 0,1,2")
    return int(surface_library().so101_surface_mode(int(mode)))


def surface_stats(reset=False):
    values = (ctypes.c_double * 7)()
    lib = surface_library()
    lib.so101_surface_stats.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.so101_surface_stats(values, int(reset))
    return list(values)


def prepare_scene(
    source_scene,
    grid_manifest,
    output_scene,
    build_dir=DEFAULT_NATIVE,
    *,
    mating_step=None,
    mating_profile="baseline",
):
    """Add grid-SDF collision geoms without altering any body/joint/inertia."""
    if mating_profile not in ("baseline", "legacy_mapped"):
        raise ValueError("Unknown mating profile")
    if mating_profile == "legacy_mapped" and mating_step is None:
        raise ValueError(
            "Choose an explicit insertion step for the candidate mating profile"
        )
    source_scene = Path(source_scene).resolve()
    grid_manifest = Path(grid_manifest).resolve()
    output_scene = Path(output_scene).resolve()
    load_backend(grid_manifest, build_dir)
    grids = json.loads(grid_manifest.read_text())["meshes"]
    root = ET.parse(source_scene).getroot()
    option = root.find("option")
    cfg = json.loads((SOURCE / "configs/so101_scene.json").read_text())
    for name, value in cfg["solver"].items():
        option.set(name, str(value))
    option.set("timestep", ".0005")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("ls_iterations", "20")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("sleep", "disable")
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("memory", cfg.get("arena_memory", "64M"))
    option.set("sdf_initpoints", str(mujoco.mjMAXCONPAIR))
    option.set("sdf_iterations", str(cfg["contact_search"]["iterations"]))
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    for name, values in [
        ("so101_sdf_search", cfg["contact_search"]),
        ("so101_assembly_search", cfg["assembly_search"]),
        ("wowrobo_grasp_search", cfg["assembly_search"]),
        ("wowrobo_hardware_search", cfg["assembly_search"]),
    ]:
        ET.SubElement(
            custom,
            "numeric",
            name=name,
            data=f"{values['initpoints']} {values['iterations']}",
        )
    ET.SubElement(custom, "numeric", name="wowrobo_surface_contact_guard", data="1")
    extension = root.find("extension")
    if extension is None:
        extension = ET.SubElement(root, "extension")
    plugin = ET.SubElement(extension, "plugin", plugin="assemblenist.sdf.grid")
    asset = root.find("asset")
    converted = []
    for body in root.iter("body"):
        for visual in list(body.findall("geom")):
            name = visual.get("name")
            if name not in grids:
                continue
            record = grids[name]
            grid_id = record.get("grid_id", record.get("id"))
            instance = name + "__grid"
            mesh = name + "__sdf_mesh"
            node = ET.SubElement(plugin, "instance", name=instance)
            ET.SubElement(node, "config", key="id", value=str(grid_id))
            mesh_node = ET.SubElement(asset, "mesh", name=mesh)
            ET.SubElement(mesh_node, "plugin", instance=instance)
            # SDF-generated meshes live in the original source-mesh coordinate
            # system. Original geom transforms are copied; no MuJoCo principal
            # mesh recenter transform is baked a second time.
            attrs = {
                k: v
                for k, v in visual.attrib.items()
                if k in ["pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"]
            }
            attrs.update(
                name=name + "__contact",
                type="sdf",
                mesh=mesh,
                contype="0",
                conaffinity="0",
                group="3",
                rgba="0 0 0 0",
                friction=".7 .005 .0001",
                solref=".002 1",
                solimp=".95 .99 .0002",
                condim="4",
            )
            geom = ET.SubElement(body, "geom", attrs)
            ET.SubElement(geom, "plugin", instance=instance)
            visual.set("contype", "0")
            visual.set("conaffinity", "0")
            converted.append(name)
    if set(converted) != set(grids):
        raise ValueError(f"Missing visual geometries: {set(grids) - set(converted)}")
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_scene, encoding="unicode")
    mating = None
    if mating_step is not None:
        from assembly_policy.so101.wowrobo_mating_profile import write_profile_scene

        baseline_scene = output_scene.with_suffix(".unmapped.xml")
        baseline_scene.write_bytes(output_scene.read_bytes())
        mating = write_profile_scene(
            baseline_scene, output_scene, mating_step, profile=mating_profile
        )
    result = {
        "scene": str(output_scene),
        "source_scene_sha256": sha256(source_scene),
        "scene_sha256": sha256(output_scene),
        "grid_manifest": str(grid_manifest),
        "grid_manifest_sha256": sha256(grid_manifest),
        "native_dir": str(Path(build_dir).resolve()),
        "contact_suffix": "__contact",
        "visual_geom_names": converted,
        "solver": dict(
            cfg["solver"],
            timestep=".0005",
            integrator="implicitfast",
            cone="elliptic",
            ls_iterations=20,
        ),
        "contact_search": cfg["contact_search"],
        "assembly_search": cfg["assembly_search"],
        "backend": "source-triangle grid SDF + bidirectional original vertices/centroids + exact accelerated surface tree",
        "scope": "Contact model only; source body/joint/inertial layout preserved; zero gravity fixture semantics retained by runner",
        "mating_profile": mating,
    }
    output_scene.with_suffix(".backend.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--grids", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    p.add_argument(
        "--mating-step", choices=["cover", "connector", "pin", "upper", "final"]
    )
    p.add_argument(
        "--mating-profile", choices=["baseline", "legacy_mapped"], default="baseline"
    )
    a = p.parse_args()
    print(
        json.dumps(
            prepare_scene(
                a.source,
                a.grids,
                a.output,
                a.native,
                mating_step=a.mating_step,
                mating_profile=a.mating_profile,
            ),
            indent=2,
        )
    )

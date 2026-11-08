"""CPU-only CAD/proxy cross sections for visual inspection of openings."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

folder = Path(__file__).resolve().parent
receipt = json.loads((folder / "candidate/receipt.json").read_text())
original_root = Path(receipt["scene"]).parent
seen = set()
for record in receipt["records"]:
    if not record["fine"] or record["key"] in seen:
        continue
    seen.add(record["key"])
    source = np.load(folder / ("source-" + record["key"][:12] + ".npz"))
    cad = trimesh.Trimesh(source["vertices"], source["faces"], process=False)
    proxies = [
        trimesh.load_mesh(
            folder / "candidate" / Path(p).relative_to(original_root), process=False
        )
        for p in record["paths"]
    ]
    lo, hi = cad.bounds
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), layout="constrained")
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for column, fraction in enumerate([0.25, 0.5, 0.75]):
            origin = (lo + hi) / 2
            origin[axis] = lo[axis] + fraction * (hi[axis] - lo[axis])
            normal = np.eye(3)[axis]
            ax = axes[axis, column]
            for mesh, color, width, alpha in [(cad, "#151515", 1.3, 1)] + [
                (h, "#e17b24", 0.8, 0.65) for h in proxies
            ]:
                lines = trimesh.intersections.mesh_plane(mesh, normal, origin)
                for line in lines:
                    ax.plot(
                        line[:, others[0]] * 1000,
                        line[:, others[1]] * 1000,
                        color=color,
                        lw=width,
                        alpha=alpha,
                    )
            ax.set_aspect("equal")
            ax.set_title(f"{'XYZ'[axis]} slice {fraction:.0%}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("mm", fontsize=8)
            ax.set_ylabel("mm", fontsize=8)
    fig.suptitle(
        record["visual"]
        + "\nBlack: actual CAD. Orange: collision proxy. Internal cut faces also shown.",
        fontsize=11,
    )
    fig.savefig(folder / ("sections-" + record["visual"] + ".png"), dpi=140)
    plt.close(fig)

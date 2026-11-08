"""Build the reproducible timing table and scientific figure from saved receipts."""

import hashlib
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mjwarp-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent


def read(name):
    f = root / (name + ".json")
    return json.loads(f.read_text()) if f.exists() else None


def metrics(data):
    if data is None:
        return None
    assert data["status"] == "PASS"
    values = [r["wall_s"] for r in data["records"]]
    return {
        "wall_s": statistics.mean(values),
        "controls_per_s": data["worlds"] / statistics.mean(values),
        "repeats": len(values),
        "spread_s": max(values) - min(values),
        "graph_ms": data["graph_ms"],
        "min_contact_distance_m": min(
            r["min_contact_distance"] for r in data["records"]
        ),
        "max_nefc": max(max(r["nefc"]) for r in data["records"]),
        "max_overflow": max(max(r["overflow"]) for r in data["records"]),
        "device_memory": data.get("device_memory"),
        "unique_states": data.get("unique_initial_states"),
        "source": data["source"],
    }


stem = "-s80-tolNone-lsNone-cap512-j2048"
table = []
for n in (16, 64, 256, 1024, 2048):
    original = "baseline-w" + str(n) + stem + ("-profile" if n == 16 else "")
    launch = (
        "baseline-w"
        + str(n)
        + stem
        + ("-profile-nb1" if n == 16 else "-profile-nb1-audit" if n == 64 else "-nb1")
    )
    final = (
        "candidate-w"
        + str(n)
        + stem
        + ("-profile-contactaudit" if n == 64 else "-selected")
    )
    table.append(
        {
            "worlds": n,
            "original": metrics(read(original)),
            "launch_only": metrics(read(launch)),
            "final": metrics(read(final)),
        }
    )
baseline = read("baseline-w16" + stem + "-profile")
scheduling = read("baseline-w16" + stem + "-profile-nb1")
kernel_total = sum(
    k["ms"] for k in baseline["kernels"] if k["name"].startswith("forward kernel")
)
noslip = sum(k["ms"] for k in baseline["kernels"] if "_noslip_solve" in k["name"])
summary = {
    "table": table,
    "baseline_noslip_kernel_fraction": noslip / kernel_total,
    "baseline_backend_mean_s": statistics.mean(
        r["wall_s"] for r in baseline["records"]
    ),
    "baseline_graph_mean_s": statistics.mean(baseline["graph_ms"]) / 1000,
}
env = read("candidate-w16" + stem + "-selected-environment-environment")
if env:
    r = env["records"][1:]
    summary["environment"] = {
        "warm_native_s": sum(x["native_s"] for x in r),
        "warm_gpu_s": sum(x["gpu_s"] for x in r),
        "native_over_gpu": sum(x["native_s"] for x in r) / sum(x["gpu_s"] for x in r),
        "max_body_error_m": max(x["max_body_position_error_m"] for x in env["records"]),
        "unsafe": any(any(x["gpu_safety"]) for x in env["records"]),
        "unstable": any(any(x["gpu_unstable"]) for x in env["records"]),
        "source": env["identity"]["implementation"],
    }
sources = Path(__file__).resolve().parents[2] / "assembly_policy/so101"
summary["final_source"] = {
    n: hashlib.sha256((sources / n).read_bytes()).hexdigest()
    for n in ("wowrobo_warp.py", "wowrobo_warp_runtime.py")
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]}
)
for key, label, color, style in [
    ("original", "Original Warp", "#939da8", "--"),
    ("launch_only", "Launch scheduling only", "#bd8f3a", ":"),
    ("final", "Selected exact optimizations", "#157e80", "-"),
]:
    rows = [r for r in table if r[key] is not None]
    left.plot(
        [r["worlds"] for r in rows],
        [r[key]["controls_per_s"] for r in rows],
        style + "o",
        color=color,
        label=label,
        lw=2,
        ms=5,
    )
    if key == "final":
        for row in rows:
            left.annotate(
                f"{row[key]['controls_per_s']:.1f}",
                (row["worlds"], row[key]["controls_per_s"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color=color,
            )
left.set_xscale("log", base=2)
left.set_xticks([16, 64, 256, 1024, 2048], ["16", "64", "256", "1,024", "2,048"])
left.set_xlabel("Worlds per batch")
left.set_ylabel("40 ms physics control intervals / second")
left.set_title(
    "Larger batches help, then returns diminish", loc="left", fontweight="bold"
)
left.grid(axis="y", alpha=0.2)
left.legend(loc="upper left", frameon=False)
names = ["No-slip solve", "Surface samples", "Ordinary SDF", "Other kernels"]
colors = ["#bd6255", "#4f86a6", "#c7a457", "#c5cdd4"]
bottom = np.zeros(2)
for cat, color in zip(names, colors):
    values = []
    for data in (baseline, scheduling):
        subtotal = {k: 0.0 for k in names}
        for item in data["kernels"]:
            if not item["name"].startswith("forward kernel"):
                continue
            name = item["name"]
            group = (
                "No-slip solve"
                if "_noslip_solve" in name
                else "Surface samples"
                if "_samples_" in name
                else "Ordinary SDF"
                if "_sdf_narrowphase_" in name
                else "Other kernels"
            )
            subtotal[group] += item["ms"] / 3
        values.append(subtotal[cat])
    right.bar([0, 1], values, bottom=bottom, color=color, label=cat, width=0.55)
    bottom += values
right.set_xticks([0, 1], ["Original", "Launch fix"])
right.set_ylabel("GPU kernel time / physics substep (ms)")
right.set_title(
    "The main bottleneck was no-slip scheduling", loc="left", fontweight="bold"
)
right.legend(frameon=False, loc="upper right", fontsize=9)
for i, total in enumerate(bottom):
    right.text(i, total + 2, f"{total:.1f} ms", ha="center")
right.set_ylim(0, max(bottom) * 1.2)
fig.suptitle(
    "Actual assembly on H100: Warp profiling and scaling",
    x=0.07,
    ha="left",
    fontweight="bold",
    fontsize=15,
)
fig.text(
    0.07,
    0.02,
    "Fixed saved-start replay • 80 × 0.5 ms substeps • unchanged geometry, friction and solver tolerances\nTwo warm repeats per point. Kernel attribution uses a separate three-substep profile at 16 worlds.",
    fontsize=8,
    color="#59616a",
)
fig.tight_layout(rect=(0, 0.10, 1, 0.94))
fig.savefig(root / "performance.png", dpi=180)
print(json.dumps(summary, indent=2))

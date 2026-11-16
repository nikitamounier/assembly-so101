"""Render saved native insertion states without changing their dynamics.

The JSON specification records model/backend identity and explicitly supplies
the captions. A replay never implies a successful insertion or retention test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from assembly_policy.so101.wowrobo_contact_backend import load_backend, set_surface_mode


def render(specification: Path) -> Path:
    spec = json.loads(specification.read_text())
    load_backend(Path(spec["grid_manifest"]))
    set_surface_mode(2)
    model = mujoco.MjModel.from_xml_path(spec["model"])
    data = mujoco.MjData(model)
    trace = np.load(spec["trace"])
    if "trace_key" in spec:
        rows = trace[spec["trace_key"]]
        times = rows[:, spec.get("time_column", 0)]
        start = spec["qpos_column"]
        positions = rows[:, start : start + model.nq]
    else:
        times = trace[spec.get("time_key", "time")]
        positions = trace[spec.get("qpos_key", "qpos")]
    if len(times) != len(positions) or positions.shape[1] != model.nq:
        raise ValueError("Trace/model shape mismatch")
    if np.any(np.diff(times) < 0) or not np.all(np.isfinite(positions)):
        raise ValueError("Invalid trace")
    output = Path(spec.get("output", str(specification.with_suffix(".mp4"))))
    output.parent.mkdir(parents=True, exist_ok=True)
    fps, width, height = 24, 960, 640
    camera = mujoco.MjvCamera()
    camera.lookat[:] = spec["lookat"]
    camera.distance = spec.get("distance", 0.38)
    camera.azimuth = spec.get("azimuth", 125)
    camera.elevation = spec.get("elevation", -24)
    options = mujoco.MjvOption()
    options.geomgroup[1] = 0
    options.geomgroup[3] = 0
    if spec.get("hide_helpers", False):
        for gid in range(model.ngeom):
            body = int(model.geom_bodyid[gid])
            while body:
                if (model.body(body).name or "").startswith(("left_", "right_")):
                    model.geom_group[gid] = 5
                    break
                body = int(model.body_parentid[body])
        options.geomgroup[5] = 0
    push = None
    if "force" in spec:
        force_spec = spec["force"]
        active = {model.body(force_spec["piece"]).id}
        for bid in range(model.nbody):
            if int(model.body_parentid[bid]) in active:
                active.add(bid)
        data.qpos[:] = positions[0]
        mujoco.mj_forward(model, data)
        push = BalancedPush(model, data, active, force_spec["axis_world"], damping=0)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(output),
    ]
    duration = float(times[-1] - times[0])
    speed = spec.get("playback_speed", 1.0)
    count = max(2, int(np.ceil((duration / speed + 1) * fps)))
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for frame in range(count):
                time = min(float(times[-1]), float(times[0]) + frame / fps * speed)
                index = min(
                    len(times) - 1, int(np.searchsorted(times, time, side="left"))
                )
                data.qpos[:] = positions[index]
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=options)
                if push is not None and times[index] < force_spec["off_s"]:
                    ramp = force_spec.get("ramp_s", 0.8)
                    magnitude = force_spec["maximum_N"] * (
                        min(1.0, float(times[index]) / ramp) if ramp > 0 else 1.0
                    )
                    points = push.apply(data, magnitude) if magnitude > 0 else []
                    axis = np.asarray(force_spec["axis_world"])
                    for point in points:
                        scene = renderer.scene
                        if scene.ngeom >= scene.maxgeom:
                            break
                        position = np.asarray(point["point_world_m"])
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            mujoco.mjtGeom.mjGEOM_ARROW,
                            np.zeros(3),
                            np.zeros(3),
                            np.eye(3).ravel(),
                            np.array([1.0, 0.55, 0.05, 1.0]),
                        )
                        mujoco.mjv_connector(
                            geom,
                            mujoco.mjtGeom.mjGEOM_ARROW,
                            0.001,
                            position - 0.018 * axis,
                            position,
                        )
                        scene.ngeom += 1
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, width, 98), fill="#f2f5f8")
                draw.text((16, 9), spec["title"], font=font, fill="#172e41")
                draw.text(
                    (16, 40),
                    f"Native trajectory replay | t={times[index]:.2f}s | MuJoCo 4.0.0",
                    font=small,
                    fill="#172e41",
                )
                draw.text((16, 69), spec["method"], font=small, fill="#713f1b")
                draw.rectangle((0, height - 62, width, height), fill="#f2f5f8")
                draw.text((16, height - 53), spec["result"], font=font, fill="#172e41")
                draw.text(
                    (16, height - 25),
                    spec.get("limitation", "Receivers fixed; isolated insertion bench"),
                    font=small,
                    fill="#713f1b",
                )
                process.stdin.write(np.asarray(image).tobytes())
                if frame in (0, count // 2, count - 1):
                    image.save(output.with_name(f"{output.stem}-{frame:03d}.png"))
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited {code}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("specification", type=Path)
    print(render(parser.parse_args().specification))

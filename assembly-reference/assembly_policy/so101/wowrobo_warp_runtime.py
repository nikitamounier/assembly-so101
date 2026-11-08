"""Opt-in WowRobo MJWarp rollout runtime (one engine per isolated process).

Implements the native Rollout interface used by the current assembly trainer.
Observations, rewards, resets and endpoint native forwards stay with the caller.
Every substep is returned for the existing penetration and velocity guards.
This is a distinct float32 dynamics backend, never an exact native continuation.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from importlib.metadata import version
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from .wowrobo_warp import WowRoboWarp

wp.set_module_options({"enable_backward": False})
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS
CONTROL = (
    mujoco.mjtState.mjSTATE_CTRL
    | mujoco.mjtState.mjSTATE_EQ_ACTIVE
    | mujoco.mjtState.mjSTATE_USERDATA
)
_active = None
HINGE_NOISE_CAP = float(np.deg2rad(0.05))


@wp.kernel
def _controls(
    values: wp.array3d[float],
    index: int,
    nu: int,
    neq: int,
    ctrl: wp.array2d[float],
    eq: wp.array2d[bool],
    userdata: wp.array2d[float],
    draws: wp.array2d[float],
):
    world, field = wp.tid()
    if field < nu:
        ctrl[world, field] = values[world, index, field]
    elif field < nu + neq:
        eq[world, field - nu] = values[world, index, field] != 0.0
    else:
        k = field - nu - neq
        userdata[world, k] = values[world, index, field]
        if k < 3:
            draws[world, k] = values[world, index, field]


@wp.kernel
def _limits(
    userdata: wp.array2d[float],
    joints: wp.array[int],
    base: wp.array[wp.vec2d],
    ranges: wp.array2d[wp.vec2],
):
    world, k = wp.tid()
    # Add in double before casting, matching native per-world range arithmetic.
    lo = base[k][0] + wp.float64(userdata[world, 3 + 2 * k])
    hi = base[k][1] + wp.float64(userdata[world, 4 + 2 * k])
    ranges[world, joints[k]] = wp.vec2(float(lo), float(hi))


@wp.kernel
def _record(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    time: wp.array[float],
    sensor: wp.array2d[float],
    states: wp.array3d[float],
    sensors: wp.array3d[float],
    index: int,
):
    world, field = wp.tid()
    if field >= states.shape[2]:
        pass
    elif field == 0:
        states[world, index, field] = time[world]
    elif field <= qpos.shape[1]:
        states[world, index, field] = qpos[world, field - 1]
    else:
        states[world, index, field] = qvel[world, field - 1 - qpos.shape[1]]
    if field < sensor.shape[1]:
        sensors[world, index, field] = sensor[world, field]


@wp.kernel
def _audit(
    nacon: wp.array[int],
    ncollision: wp.array[int],
    nefc: wp.array[int],
    overflow: wp.array[int],
    errors: wp.array[int],
    capacity: int,
    njmax: int,
    flags: wp.array[int],
):
    # Latch transient failures: the engine resets these fields at the next step.
    if nacon[0] > capacity or ncollision[0] > capacity or errors[0] != 0:
        wp.atomic_max(flags, 0, 1)
    for world in range(nefc.shape[0]):
        if nefc[world] > njmax:
            wp.atomic_max(flags, 0, 2)
    for i in range(overflow.shape[0]):
        if overflow[i] != 0:
            wp.atomic_max(flags, 0, 3)


class WowRoboWarpRollout:
    """Batched CPU-array interface with reusable CUDA graphs and explicit identity.

    Retains the supplied model's solver settings. Per-world hinge range noise
    uses MJWarp's batched jnt_range array; shared native model arrays never change.
    Pass exactly CTRL|EQ_ACTIVE|USERDATA, and initial_warmstart when resetting.
    """

    def __init__(
        self,
        native,
        grid_directory,
        scratch,
        *,
        worlds,
        device="cuda:0",
        jacobian="dense",
        nconmax=512,
        njmax=2048,
        hinge_noise_cap=HINGE_NOISE_CAP,
        noslip_block_dim=1,
    ):
        global _active
        if _active is not None:
            raise RuntimeError("Use a separate process for each WowRobo Warp runtime")
        versions = {n: version(n) for n in ("mujoco", "mujoco-warp", "warp-lang")}
        if versions["mujoco"] != "4.0.0" or versions["mujoco-warp"] != "4.0.0":
            raise RuntimeError(
                "This private contact port requires MuJoCo and MJWarp 4.0.0"
            )
        if versions["warp-lang"].split("+")[0] != "1.19.0":
            raise RuntimeError("This private contact port requires Warp 1.19.0")
        if (
            worlds < 1
            or native.na
            or native.npluginstate
            or native.nmocap
            or native.nhistory
        ):
            raise ValueError(
                "Expected positive worlds, no act/plugin/history/mocap state"
            )
        if native.nuserdata not in (3, 7):
            raise ValueError(
                "Expected three friction draws and optionally four hinge offsets"
            )
        if not 0 <= hinge_noise_cap < 0.01:
            raise ValueError("Invalid hinge noise cap")
        if set(map(int, native.geom_plugin[native.geom_plugin >= 0])) != set(
            range(native.nplugin)
        ):
            raise ValueError("Non-geometry plugin callbacks have no GPU mapping")
        if any(native.plugin_stateadr >= 0) and native.npluginstate:
            raise ValueError("Stateful plugins have no GPU mapping")
        self.native, self.worlds = native, int(worlds)
        self.full_size = mujoco.mj_stateSize(native, STATE)
        if self.full_size != 1 + native.nq + native.nv:
            raise ValueError("Unmapped FULLPHYSICS state")
        self.ncontrol = native.nu + native.neq + native.nuserdata
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("WowRobo Warp runtime requires CUDA")
        self.hinge_noise_cap = hinge_noise_cap
        self.closed = False
        self.buffers = {}
        self.warmed = False
        self.calls = 0
        self.physics = None
        self._noslip_patch = None
        self.hinge_ids = []
        _active = self
        try:
            if noslip_block_dim is not None:
                self._install_noslip_schedule(scratch, noslip_block_dim)
            with wp.ScopedDevice(self.device):
                self.physics = WowRoboWarp(
                    native,
                    mujoco.MjData(native),
                    grid_directory,
                    scratch,
                    nworld=worlds,
                    nconmax=nconmax,
                    njmax=njmax,
                    jacobian=jacobian,
                )
                p = self.physics
                p.install_context_capture()
                if native.nuserdata == 7:
                    self.hinge_ids = [
                        native.joint(n).id
                        for n in ("target_servo_2_shaft", "target_servo_3_shaft")
                    ]
                    if not np.all(native.jnt_limited[self.hinge_ids]):
                        raise ValueError("Noisy hinges must have measured limits")
                    self.hinge_joints = wp.array(self.hinge_ids, dtype=int)
                    self.hinge_base = wp.array(
                        native.jnt_range[self.hinge_ids], dtype=wp.vec2d
                    )
                    p.model.jnt_range = wp.array(
                        np.tile(native.jnt_range, (worlds, 1, 1)), dtype=wp.vec2
                    )
                self.flags = wp.zeros(1, dtype=int)
            self.identity = {
                "backend": "wowrobo-mjwarp-runtime-v1",
                "versions": versions,
                "worlds": worlds,
                "jacobian": jacobian,
                "nconmax": nconmax,
                "njmax": njmax,
                "timestep": float(native.opt.timestep),
                "tolerance": float(native.opt.tolerance),
                "iterations": int(native.opt.iterations),
                "state_precision": "float32",
                "sdf_precision": "float64",
                "hinge_noise_cap": hinge_noise_cap,
                "qualified_for_training": False,
                "noslip_block_dim": noslip_block_dim,
                "implementation": {
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in (
                        Path(__file__),
                        Path(__file__).with_name("wowrobo_warp.py"),
                    )
                },
            }
            self.identity["native_sources"] = {
                key: hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                for key in ("WOWROBO_WARP_GRID_SOURCE", "WOWROBO_WARP_FRICTION_SOURCE")
                if key in os.environ
            }
        except BaseException:
            self.close()
            raise

    def _install_noslip_schedule(self, scratch, block_dim):
        """Spread independent serial world solves across SMs; retain kernel math.

        Compile only the launch function in a private namespace. Cached upstream
        kernels, model/iteration settings and all within-world operations remain
        unchanged. Installed package files are never edited.
        """
        from mujoco_warp._src import solver

        if block_dim not in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError("Unsupported noslip launch block size")
        original = solver.noslip
        source = inspect.getsource(original)
        anchor = "    _noslip_solve(m.is_sparse, sleep),\n    dim=d.nworld,"
        if source.count(anchor) != 1:
            raise RuntimeError("Installed noslip launch differs from the pinned API")
        source = source.replace(anchor, anchor + f"\n    block_dim={block_dim},")
        path = Path(scratch) / "private_noslip_schedule.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        namespace = dict(solver.__dict__)
        exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 - pinned installed function, launch-only patch
        patched = namespace["noslip"]
        solver.noslip = patched
        self._noslip_patch = (solver, original, patched)

    def _validate(self, initial_state, control, nstep, initial_warmstart):
        m = self.native
        state, control = np.asarray(initial_state, float), np.asarray(control, float)
        warm = (
            np.zeros((self.worlds, m.nv))
            if initial_warmstart is None
            else np.asarray(initial_warmstart, float)
        )
        if state.shape != (self.worlds, self.full_size) or warm.shape != (
            self.worlds,
            m.nv,
        ):
            raise ValueError("Invalid FULLPHYSICS/warmstart batch shape")
        if nstep < 1 or control.shape != (self.worlds, nstep, self.ncontrol):
            raise ValueError("Invalid CTRL|EQ_ACTIVE|USERDATA control shape")
        if not all(np.isfinite(x).all() for x in (state, control, warm)):
            raise ValueError("Nonfinite rollout input")
        eq = control[:, :, m.nu : m.nu + m.neq]
        if not np.isin(eq, [0, 1]).all():
            raise ValueError("Equality activation must be binary")
        userdata = control[:, :, m.nu + m.neq :]
        if (userdata[:, :, :3] <= 0).any():
            raise ValueError("Friction draws must be positive")
        if self.hinge_ids:
            noise = userdata[:, :, 3:].reshape(self.worlds, nstep, 2, 2)
            if (np.abs(noise) > self.hinge_noise_cap).any():
                raise ValueError("Hinge noise exceeds the native contract")
            bounds = m.jnt_range[self.hinge_ids] + noise
            if (bounds[:, :, :, 0] >= bounds[:, :, :, 1]).any():
                raise ValueError("Inverted per-world hinge range")
        # Check overflow caused by float32 conversion before device upload.
        with np.errstate(over="ignore"):
            if not all(
                np.isfinite(x.astype(np.float32)).all() for x in (state, control, warm)
            ):
                raise ValueError("Rollout input exceeds float32 range")
        return state, control, warm

    def _restore(self, state, warm):
        p, m = self.physics, self.native
        mjw.reset_data(p.model, p.data)
        p.data.time.assign(state[:, 0].astype(np.float32))
        p.data.qpos.assign(state[:, 1 : 1 + m.nq].astype(np.float32))
        p.data.qvel.assign(state[:, 1 + m.nq :].astype(np.float32))
        p.data.qacc_warmstart.assign(warm.astype(np.float32))
        self.flags.zero_()

    def _step(self, inputs, states, sensors, index):
        p, m = self.physics, self.native
        wp.launch(
            _controls,
            dim=(self.worlds, self.ncontrol),
            inputs=[
                inputs,
                index,
                m.nu,
                m.neq,
                p.data.ctrl,
                p.data.eq_active,
                p.data.userdata,
                p.draws,
            ],
        )
        if self.hinge_ids:
            wp.launch(
                _limits,
                dim=(self.worlds, 2),
                inputs=[
                    p.data.userdata,
                    self.hinge_joints,
                    self.hinge_base,
                    p.model.jnt_range,
                ],
            )
        mjw.step(p.model, p.data)
        wp.launch(
            _record,
            dim=(self.worlds, max(self.full_size, m.nsensordata)),
            inputs=[
                p.data.qpos,
                p.data.qvel,
                p.data.time,
                p.data.sensordata,
                states,
                sensors,
                index,
            ],
        )
        wp.launch(
            _audit,
            dim=1,
            inputs=[
                p.data.nacon,
                p.data.ncollision,
                p.data.nefc,
                p.data.overflow,
                p.errors,
                p.data.naconmax,
                p.data.njmax,
                self.flags,
            ],
        )

    def rollout(
        self,
        model,
        work,
        initial_state,
        control,
        *,
        nstep,
        control_spec=CONTROL,
        initial_warmstart=None,
    ):
        if self.closed or model is not self.native:
            raise ValueError("Closed runtime or changed model")
        if int(control_spec) != int(CONTROL):
            raise ValueError("Expected CTRL|EQ_ACTIVE|USERDATA")
        state, control, warm = self._validate(
            initial_state, control, nstep, initial_warmstart
        )
        with wp.ScopedDevice(self.device):
            self._restore(state, warm)
            if nstep not in self.buffers:
                inputs = wp.array(control.astype(np.float32), dtype=float)
                states = wp.empty((self.worlds, nstep, self.full_size), dtype=float)
                sensors = wp.empty((self.worlds, nstep, model.nsensordata), dtype=float)
                # Compile outside capture. Discard the warmup integration state.
                self._step(inputs, states, sensors, 0)
                wp.synchronize_device(self.device)
                self._restore(state, warm)
                with wp.ScopedCapture(device=self.device) as capture:
                    for index in range(nstep):
                        self._step(inputs, states, sensors, index)
                self.buffers[nstep] = (inputs, states, sensors, capture.graph)
                # Capture may execute on some Warp versions; restore either way.
                self._restore(state, warm)
            inputs, states, sensors, graph = self.buffers[nstep]
            inputs.assign(control.astype(np.float32))
            wp.capture_launch(graph)
            raw, readings, flags = states.numpy(), sensors.numpy(), self.flags.numpy()
        if flags.any():
            raise FloatingPointError(
                f"MJWarp transient capacity/convergence failure: {flags.tolist()}"
            )
        if not np.isfinite(raw).all() or not np.isfinite(readings).all():
            raise FloatingPointError("Nonfinite MJWarp trajectory or sensor")
        expected = np.empty((self.worlds, nstep), np.float32)
        clock = state[:, 0].astype(np.float32)
        for index in range(nstep):
            clock = clock + np.float32(model.opt.timestep)
            expected[:, index] = clock
        if not np.array_equal(raw[:, :, 0], expected):
            raise FloatingPointError("MJWarp integration clock diverged or reset")
        result = raw.astype(np.float64)
        result[:, :, 0] = (
            state[:, None, 0] + np.arange(1, nstep + 1) * model.opt.timestep
        )
        self.calls += 1
        return result, readings.astype(np.float64)

    def close(self):
        global _active
        if not self.closed:
            if self.physics is not None:
                with wp.ScopedDevice(self.device):
                    self.physics.close()
            self.buffers.clear()
            if self._noslip_patch is not None:
                solver, original, patched = self._noslip_patch
                if solver.noslip is patched:
                    solver.noslip = original
            self.closed = True
        if _active is self:
            _active = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def install_on_environment(env, scratch, *, grid_directory=None, **options):
    """Explicitly replace an initialized current environment's rollout engine.

    Call before its first step in a separate run/process. The existing native
    environment still owns all state, rewards, observations and reset scheduling.
    Persist env.warp_identity in the run config; never label it native physics.
    """
    import json

    if not getattr(env, "extra_reset", False) or not getattr(
        env, "episode_friction", False
    ):
        raise ValueError(
            "Current fullphysics/eq/ctrl/warmstart/friction environment required"
        )
    if grid_directory is None:
        manifest = Path(env.profile["grid_manifest"])
        meta = json.loads(manifest.read_text())
        grid_directory = Path(
            meta.get("grid_dir", meta.get("grid_directory", manifest.parent))
        )
        if not grid_directory.is_absolute():
            grid_directory = manifest.parent / grid_directory
    options.setdefault(
        "hinge_noise_cap", env.profile.get("hinge_limit_noise_rad", HINGE_NOISE_CAP)
    )
    engine = WowRoboWarpRollout(env.m, grid_directory, scratch, worlds=env.N, **options)
    env.engine.close()
    env.engine = engine
    env.warp_identity = dict(
        engine.identity,
        native_model_signature=env.model_signature,
        profile_sha256=hashlib.sha256(
            Path(env.ecfg.profile_path).read_bytes()
        ).hexdigest(),
    )
    return env

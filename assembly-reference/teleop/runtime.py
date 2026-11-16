"""Local synthetic MuJoCo preview; no hardware or remote timing qualification.

Request threads exchange immutable messages/snapshots. Only the 50 Hz owner
touches the controller/plant; only the main thread touches the renderer.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler
import io
import json
import math
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit

import mujoco
import numpy as np
from PIL import Image

from .control import SimulationController, rotation

MAX_AGE_NS = 200_000_000
FRAME_AGE_NS = 500_000_000
PERIOD_NS = 20_000_000
SIDES = ("left", "right")
PHONE_JPEG_BYTES = 1_048_576
PHONE_BODY_BYTES = 1_572_864
PHONE_WIDTH, PHONE_HEIGHT = 1920, 1080


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite_number_required")
    return float(value)


def json_bytes(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def pose_json(pose):
    return {"position": pose.position.tolist(), "rotation": pose.rotation.reshape(-1).tolist()}


class PhoneRelay:
    """One bounded video-only source in memory; never a control session/journal."""
    def __init__(self, clock=time.monotonic_ns):
        self.clock = clock
        self.lock = threading.Lock()
        self.generation = 0
        self.session = None
        self.frame = None

    def create_session(self, client_ms):
        c0 = finite(client_ms)
        if c0 < 0:
            raise ValueError("negative_client_clock")
        now = self.clock()
        with self.lock:
            self.generation += 1
            self.frame = None
            self.session = dict(identity=secrets.token_urlsafe(24), c0=c0, s1=now, c2=None)
            return dict(session=self.session["identity"], client_ms=c0, server_ms=now/1e6, generation=self.generation)

    def receive_frame(self, raw):
        with self.lock:
            try:
                if self.session is None or raw.get("session") != self.session["identity"]:
                    raise ValueError("unknown_or_superseded_phone_session")
                sequence = raw.get("sequence")
                if type(sequence) is not int or sequence < 0 or (self.frame and sequence <= self.frame["sequence"]):
                    raise ValueError("reordered_phone_sequence")
                source, c2, media = (finite(raw.get(k)) for k in ("decoded_ms", "sync_receive_ms", "media_time_s"))
                if c2 < self.session["c0"] or source < c2 or media < 0:
                    raise ValueError("invalid_phone_clock_or_media_time")
                if self.session["c2"] is not None and c2 != self.session["c2"]:
                    raise ValueError("phone_clock_binding_changed")
                if self.frame and (source <= self.frame["decoded_ms"] or media <= self.frame["media_time_s"]):
                    raise ValueError("reordered_phone_original_decode_or_media")
                source_ns = now if self.receipt_timing else math.floor(source*1e6 + self.session["s1"] - c2*1e6)
                outcome["age_basis"] = "server_receipt" if self.receipt_timing else "original_source"
                if not 0 <= self.clock()-source_ns < FRAME_AGE_NS:
                    raise ValueError("future_or_expired_phone_original_decode")
                width, height = raw.get("width"), raw.get("height")
                if (type(width) is not int or type(height) is not int
                        or not 0 < width <= PHONE_WIDTH or not 0 < height <= PHONE_HEIGHT):
                    raise ValueError("bounded_phone_dimensions_required")
                if raw.get("sensor_time_known") is not False:
                    raise ValueError("phone_sensor_time_must_be_unknown")
                jpeg = raw.get("jpeg")
                prefix = "data:image/jpeg;base64,"
                if not isinstance(jpeg, str) or not jpeg.startswith(prefix) or len(jpeg) > len(prefix)+4*((PHONE_JPEG_BYTES+2)//3):
                    raise ValueError("bounded_jpeg_data_url_required")
                payload = base64.b64decode(jpeg[len(prefix):], validate=True)
                if not payload or len(payload) > PHONE_JPEG_BYTES:
                    raise ValueError("bounded_jpeg_bytes_required")
                with Image.open(io.BytesIO(payload)) as image:
                    if image.format != "JPEG" or image.size != (width, height):
                        raise ValueError("phone_jpeg_dimensions_or_format_mismatch")
                    image.load()
                now = self.clock()
                if not 0 <= now-source_ns < FRAME_AGE_NS:
                    raise ValueError("phone_decode_expired_during_validation")
                self.session["c2"] = c2
                self.frame = dict(sequence=sequence, decoded_ms=source, decoded_ns=source_ns,
                                  media_time_s=media, width=width, height=height, jpeg=jpeg)
                return dict(accepted=True, reason="latest_phone_frame", generation=self.generation,
                            sequence=sequence, age_upper_ms=(now-source_ns)/1e6, decoded_server_ms=source_ns/1e6)
            except (ValueError, TypeError, OverflowError, OSError) as error:
                return dict(accepted=False, reason=str(error), generation=self.generation)

    def stop(self, identity):
        with self.lock:
            if self.session is None or identity != self.session["identity"]:
                return dict(accepted=False, reason="unknown_or_superseded_phone_session", generation=self.generation)
            self.session = self.frame = None
            self.generation += 1
            return dict(accepted=True, reason="phone_publisher_stopped", generation=self.generation)

    def get(self):
        with self.lock:
            frame = self.frame
            age = None if frame is None else (self.clock()-frame["decoded_ns"])/1e6
            available = frame is not None and 0 <= age < FRAME_AGE_NS/1e6
            result = dict(available=available, generation=self.generation,
                          sequence=-1 if frame is None else frame["sequence"],
                          decoded_server_ms=None if frame is None else frame["decoded_ns"]/1e6,
                          age_upper_ms=age, sensor_time_known=False,
                          reason="latest_phone_frame" if available else "phone_missing_or_original_decode_expired")
            if available:
                result.update(width=frame["width"], height=frame["height"], media_time_s=frame["media_time_s"], jpeg=frame["jpeg"])
            return result


class Journal:
    """A new directory per run, retaining raw timestamps and source identities."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.files = {name: (self.directory / (name + ".jsonl")).open("x") for name in ("input", "state", "events")}
        from assembly_policy.so101.scene import ROOT
        paths = [Path(__file__), Path(__file__).with_name('control.py'), Path(__file__).with_name('backend.py'),
                 ROOT/'assembly_policy/so101/scene.py', ROOT/'assembly_policy/so101/workcell.py', ROOT/'configs/so101_scene.json']
        binding = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
        (self.directory / "source-binding.json").write_bytes(json_bytes(binding) + b"\n")

    def write(self, name, value):
        line = json_bytes(value).decode() + "\n"
        with self.lock:
            self.files[name].write(line)
            self.files[name].flush()

    def close(self):
        with self.lock:
            for stream in self.files.values():
                stream.close()


@dataclass(frozen=True)
class Input:
    session: str
    sequence: int
    source_ns: int
    source_ms: float
    controllers: dict
    actions: dict
    focus: bool
    recovery_held: bool = False


@dataclass(frozen=True)
class Capture:
    qpos: tuple
    acquired_ns: int
    state_json: bytes

    @property
    def state(self):
        return json.loads(self.state_json)


class PreviewRuntime:
    def __init__(self, model, description, *, controller_factory=SimulationController,
                 clock=time.monotonic_ns, journal=None, allow_camera_fixture=False, receipt_timing=False, require_both_controllers=True):
        self.model, self.description = model, description
        self.receipt_timing = bool(receipt_timing)
        self.require_both_controllers = bool(require_both_controllers)
        self.owner_id = secrets.token_hex(12)  # public process identity, not a control token
        self.controller_factory, self.clock, self.journal = controller_factory, clock, journal
        self.phone = PhoneRelay(clock)
        self.cameras = dict(description.get("cameras", {}))
        if set(self.cameras) != {"main", "left", "right"}:
            if not allow_camera_fixture:
                raise ValueError("three_native_cameras_required: integrate the camera model first")
            self.cameras = {"main": description.get("camera", "workspace_camera")}
        self.camera = description.get("main_camera")
        if self.camera is None:
            if not allow_camera_fixture:
                raise ValueError("main_camera_pose_and_nominal_intrinsics_required")
            self.camera = dict(position=[0., 0., 0.], rotation=np.eye(3).ravel().tolist(), fovy_degrees=60., aspect=16/9)
        self.camera = dict(self.camera)
        self.camera["rotation"] = rotation(np.asarray(self.camera["rotation"], dtype=float).reshape(3, 3)).ravel().tolist()
        json_bytes(self.camera)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.owner_thread = None
        self.owner_ident = None
        self.controller = None
        self.session = None
        self.slot = None
        self.pending_hold = None
        self.pending_stop = False
        self.pending_disable = False
        self.pending_lifecycle = {}
        self.received_actions = {}
        self.received_recovery_held = False
        self.pending_recovery_release = None
        self.session_changed = False
        self.frames = {}
        self.frame_acquired_ns = None
        self.frame_sequence = 0
        self.frame_state_json = None
        self.robot_overlay = None
        self.owner_acquired_ns = None
        self.snapshot = None
        self.owner_error = None
        self.render_error = None
        self.render_alive = False
        self.last_grip = {s: True for s in SIDES}
        self.recovery_ready = False
        self.recovery_last_ns = None
        self.torque_hold_source_ns = None
        self.grip_ready = {s: False for s in SIDES}
        self.accepted_sequence = -1
        self.physics_steps_per_cycle = 10
        self.input_poll_steps = 0
        self.state = dict(owner_id=self.owner_id, sequence=-1, simulation_s=0., enabled=False, stopped=False,
                          reason="owner_starting", faults=[], tools={}, goals={}, grabs={}, gain=1., server_ms=clock()/1e6)

    def _record(self, name, value):
        if self.journal:
            self.journal.write(name, value)

    def create_session(self, client_ms):
        c0 = finite(client_ms)
        if c0 < 0:
            raise ValueError("negative_client_clock")
        now = self.clock()
        with self.lock:
            identity = secrets.token_urlsafe(24)
            self.session = dict(identity=identity, c0=c0, s1=now, c2=None, sequence=-1, source_ms=None)
            self.slot = None
            self.pending_lifecycle.clear()
            self.received_actions = {}
            self.received_recovery_held = False
            self.pending_recovery_release = None
            self.session_changed = True
        self._record("events", dict(event="session_disabled", session=identity, client_ms=c0, server_ns=now))
        return dict(session=identity, owner_id=self.owner_id, server_ms=now/1e6, client_ms=c0)

    def receive_input(self, raw):
        now = self.clock()
        outcome = dict(accepted=False, reason="invalid_input", owner_id=self.owner_id, server_ms=now/1e6)
        with self.lock:
            matching = isinstance(raw, dict) and self.session is not None and raw.get("session") == self.session["identity"]
            try:
                if not matching:
                    raise ValueError("unknown_or_superseded_session")
                actions = raw.get("actions")
                # A valid-session Stop survives malformed, old and overwritten input.
                if isinstance(actions, dict) and actions.get("stop") is True:
                    self.pending_stop = True
                    self.pending_lifecycle.clear()
                seq = raw.get("sequence")
                if type(seq) is not int or seq < 0 or seq <= self.session["sequence"]:
                    raise ValueError("reordered_sequence")
                self.session["sequence"] = seq
                outcome["sequence"] = seq
                source = finite(raw.get("source_ms"))
                c2 = finite(raw.get("sync_receive_ms"))
                if c2 < self.session["c0"] or source < c2:
                    raise ValueError("invalid_clock_binding_or_source_before_sync")
                if self.session["c2"] is None:
                    self.session["c2"] = c2
                elif c2 != self.session["c2"]:
                    raise ValueError("session_clock_binding_changed")
                previous = self.session["source_ms"]
                if previous is not None and source <= previous:
                    raise ValueError("reordered_original_source")
                source_ns = now if self.receipt_timing else math.floor(source*1e6 + self.session["s1"] - c2*1e6)
                outcome["age_basis"] = "server_receipt" if self.receipt_timing else "original_source"
                outcome["earliest_source_ns"] = source_ns
                outcome["upper_age_ms"] = (now-source_ns)/1e6
                if not 0 <= now-source_ns < MAX_AGE_NS:
                    raise ValueError("future_or_expired_original_source")
                if type(raw.get("focus")) is not bool or not isinstance(actions, dict):
                    raise ValueError("focus_and_actions_required")
                if any(type(actions.get(k, False)) is not bool for k in ("enable", "disable", "stop", "reset", "torque_release", "environment_reset", "advance_task")):
                    raise ValueError("boolean_actions_required")
                if type(raw.get('recovery_held', False)) is not bool:
                    raise ValueError('boolean_recovery_held_required')
                controllers = raw.get("controllers")
                if not isinstance(controllers, dict):
                    raise ValueError("controllers_required")
                parsed = {}
                for side in self.description["arms"]:
                    c = controllers.get(side)
                    if not isinstance(c, dict) or type(c.get("tracked")) is not bool:
                        raise ValueError("tracked_boolean_required_" + side)
                    if not c["tracked"]:
                        parsed[side] = dict(tracked=False, grip=False)
                        continue
                    if type(c.get("grip")) is not bool:
                        raise ValueError("grip_boolean_required_" + side)
                    p = np.asarray(c.get("position"), dtype=float)
                    if p.shape != (3,) or not np.isfinite(p).all():
                        raise ValueError("finite_position_required_" + side)
                    r = np.asarray(c.get("rotation"), dtype=float)
                    if r.shape != (9,):
                        raise ValueError("row_major_rotation_required_" + side)
                    r = rotation(r.reshape(3, 3))
                    trigger = finite(c.get("trigger"))
                    if not 0 <= trigger <= 1:
                        raise ValueError("trigger_range_required_" + side)
                    parsed[side] = dict(tracked=True, grip=c["grip"], position=p.copy(), rotation=r, trigger=trigger)
                self.session["source_ms"] = source
                self.pending_disable |= actions.get("disable", False)
                self.slot = Input(self.session["identity"], seq, source_ns, source, parsed,
                                  {k: actions.get(k, False) for k in ("enable", "disable", "stop", "reset", "torque_release", "environment_reset", "advance_task")}, raw["focus"], raw.get("recovery_held", False))
                if self.received_recovery_held and not self.slot.recovery_held:
                    self.pending_recovery_release = self.slot
                self.received_recovery_held = self.slot.recovery_held
                if self.pending_stop or self.pending_disable or self.pending_hold:
                    self.pending_lifecycle.clear()
                else:
                    for action in ("enable", "reset", "environment_reset", "advance_task"):
                        if actions.get(action, False) and not self.received_actions.get(action, False):
                            self.pending_lifecycle.setdefault(action, self.slot)
                self.received_actions = self.slot.actions.copy()
                outcome.update(accepted=True, reason="queued_latest_original_sample")
            except (ValueError, TypeError, OverflowError) as error:
                outcome["reason"] = str(error)
                if matching:
                    self.pending_hold = "input_rejected:" + str(error)
                    self.slot = None
                    self.pending_lifecycle.clear()
        # Invalid JSON/NaN is rejected before this method by HTTP; valid raw
        # source fields remain exactly as supplied in the evidence journal.
        self._record("input", dict(received_ns=now, original=raw, outcome=outcome))
        return outcome

    def reject_malformed_input(self, reason, original_text):
        with self.lock:
            self.pending_hold = "input_rejected:" + reason
            self.slot = None
            self.pending_lifecycle.clear()
        self._record("input", dict(received_ns=self.clock(), original_text=original_text,
                                   outcome=dict(accepted=False, reason=reason)))

    def initialize_owner(self):
        if self.owner_ident is not None:
            raise RuntimeError("owner_already_initialized")
        self.owner_ident = threading.get_ident()
        self.controller = self.controller_factory(self.model, self.description)
        if self.controller.mapper.gain != 1.:
            raise ValueError("gain1_controller_required: integrate the selected controller")
        self.publish(self.clock())

    def _owned(self):
        if threading.get_ident() != self.owner_ident:
            raise RuntimeError("controller_owner_thread_required")

    def _fresh(self, now):
        with self.lock:
            return (self.owner_acquired_ns is not None and 0 <= now-self.owner_acquired_ns < MAX_AGE_NS
                    and self.frame_acquired_ns is not None and 0 <= now-self.frame_acquired_ns < FRAME_AGE_NS
                    and set(self.frames) == set(self.cameras) and not self.render_error)

    def cycle(self, now=None, *, advance_physics=True):
        self._owned()
        now = self.clock() if now is None else now
        ctl = self.controller
        # A missed original Y deadline consumes the gesture. Fresh held samples
        # cannot repair it until the owner observes Y released again.
        if self.recovery_last_ns is not None and not 0 <= now-self.recovery_last_ns < MAX_AGE_NS:
            self.recovery_ready = False
            self.recovery_last_ns = None
            if ctl.recovery_active:
                ctl.enabled = False
                ctl.hold('original_recovery_input_expired')
            else:ctl.set_recovery(False, now, now)
        with self.lock:
            packet, self.slot = self.slot, None
            changed, self.session_changed = self.session_changed, False
            hold, self.pending_hold = self.pending_hold, None
            stop, self.pending_stop = self.pending_stop, False
            disable, self.pending_disable = self.pending_disable, False
            lifecycle, self.pending_lifecycle = self.pending_lifecycle, {}
            y_release, self.pending_recovery_release = self.pending_recovery_release, None
        if y_release is not None:
            self.torque_hold_source_ns = None
            ctl.set_recovery(False, y_release.source_ns, now)
            self.recovery_last_ns = None
            self.recovery_ready = (0 <= now-y_release.source_ns < MAX_AGE_NS and y_release.focus
                                   and all(c['tracked'] for c in y_release.controllers.values()))
        if changed:
            ctl.submit_status.clear()
            ctl.enabled = False
            ctl.hold("new_session_disabled")
            self.last_grip = {s: True for s in SIDES}
            self.recovery_ready = False
            self.recovery_last_ns = None
            self.grip_ready = {s: False for s in SIDES}
        if hold:
            ctl.hold(hold)
            self.recovery_ready = False
            self.recovery_last_ns = None
            self.grip_ready = {s: False for s in SIDES}
        if stop:
            ctl.stop()
            packet = None
            self.recovery_ready = False
            self.recovery_last_ns = None
            self.grip_ready = {s: False for s in SIDES}
        elif disable:
            self.recovery_ready = False
            self.recovery_last_ns = None
            ctl.enabled = False
            ctl.hold("disabled_layout")
        fresh = self._fresh(now)
        if (ctl.enabled or ctl.recovery_source_ns is not None) and not fresh:
            ctl.hold("owner_or_frame_original_acquisition_expired")
            self.recovery_ready = False
            self.recovery_last_ns = None
            self.grip_ready = {s: False for s in SIDES}
        if packet is not None:
            with self.lock:
                current = self.session and packet.session == self.session["identity"]
            if not current or not 0 <= now-packet.source_ns < MAX_AGE_NS:
                ctl.hold("input_expired_before_owner")
                self.recovery_ready = False
                self.recovery_last_ns = None
                self.grip_ready = {s: False for s in SIDES}
            else:
                scene_action = next((a for a in ("environment_reset", "advance_task") if a in lifecycle), None)
                event = lifecycle.get(scene_action) or lifecycle.get("reset") or lifecycle.get("enable")
                event_fresh = event is not None and event.session == packet.session and 0 <= now-event.source_ns < MAX_AGE_NS
                released = all(not c["grip"] for c in packet.controllers.values())
                if disable:
                    ctl.enabled = False
                    ctl.hold("disabled_layout")
                elif hold:
                    pass  # Rejection/Hold cancels pending authority for this owner cycle.
                elif event is not None and not event_fresh:
                    ctl.hold("lifecycle_input_expired_before_owner")
                    self.recovery_ready = False
                    self.recovery_last_ns = None
                    self.grip_ready = {s: False for s in SIDES}
                elif scene_action:
                    if event.focus and packet.focus and fresh:
                        self.apply_scene_action(scene_action)
                        ctl = self.controller
                        self.recovery_ready = False
                        self.recovery_last_ns = None
                        self.grip_ready = {s: False for s in SIDES}
                    else:
                        ctl.hold("scene_change_requires_fresh_focused_session")
                elif "reset" in lifecycle:
                    event_released = all(not c["grip"] for c in event.controllers.values())
                    if ctl.stopped and released and event_released and not packet.actions["stop"] and ctl.reset_stop():
                        ctl.enabled = False
                        self.recovery_ready = False
                        self.recovery_last_ns = None
                        self.grip_ready = {s: False for s in SIDES}
                    elif ctl.stopped:
                        ctl.reason = "reset_requires_released_grips_and_observed_rest"
                elif "enable" in lifecycle:
                    if (event.focus and packet.focus and fresh
                            and (all if self.require_both_controllers else any)(c["tracked"] for c in event.controllers.values())
                            and (all if self.require_both_controllers else any)(c["tracked"] for c in packet.controllers.values())):
                        if ctl.enable():
                            self.grip_ready = {s: not c["grip"] and not event.controllers[s]["grip"] for s, c in packet.controllers.items()}
                    else:
                        ctl.reason = "enable_requires_fresh_owner_frames_focus_and_tracking"
                if not packet.focus:
                    ctl.hold("focus_lost")
                    self.recovery_ready = False
                    self.recovery_last_ns = None
                    self.grip_ready = {s: False for s in SIDES}
                recovery_allowed = (not disable and not hold and 'reset' not in lifecycle and packet.focus and fresh
                                    and all(c['tracked'] for c in packet.controllers.values()) and not ctl.stopped)
                if not recovery_allowed:self.recovery_ready = False
                elif not packet.recovery_held:self.recovery_ready = True
                admitted_Y = packet.recovery_held and recovery_allowed and self.recovery_ready
                if not admitted_Y or not released:
                    self.torque_hold_source_ns = None
                elif self.recovery_last_ns is None:
                    self.torque_hold_source_ns = packet.source_ns if not ctl.enabled else None
                ctl.set_recovery(admitted_Y, packet.source_ns, now)
                self.recovery_last_ns = packet.source_ns if admitted_Y else None
                for side, c in packet.controllers.items():
                    grip = c["grip"] and c["tracked"] and packet.focus
                    if not grip:
                        if side in ctl.mapper.grabs or side in ctl.samples or side in ctl.goals:
                            ctl.release(side)
                        self.grip_ready[side] = c["tracked"] and packet.focus
                    elif (ctl.enabled or ctl.recovery_held(now)) and not ctl.stopped and fresh:
                        if not self.last_grip[side] and self.grip_ready[side] and side not in ctl.mapper.grabs:
                            ctl.grab(side, c["position"], c["rotation"], packet.source_ns, now)
                        if side in ctl.mapper.grabs:
                            if ctl.submit(side, c["position"], c["rotation"], packet.source_ns, now):
                                ctl.gripper(side, 1.-c["trigger"], now)
                            else:
                                self.grip_ready[side] = False
                    self.last_grip[side] = grip
                if packet.actions['torque_release']:
                    if (admitted_Y and released and self.torque_hold_source_ns is not None
                            and packet.source_ns-self.torque_hold_source_ns >= 3_000_000_000
                            and not ctl.enabled and not ctl.stopped and fresh):
                        ctl.release_torque()
                    else:
                        ctl.reason = 'torque_release_requires_disabled_fresh_3s_Y_and_released_grips'
                self.accepted_sequence = packet.sequence
        if advance_physics:
            for step in range(self.physics_steps_per_cycle):
                if step and self.input_poll_steps and step % self.input_poll_steps == 0:
                    with self.lock:
                        waiting = (self.slot is not None or self.pending_stop or self.pending_disable or self.pending_hold)
                    if waiting:
                        # Admit genuinely new receipts between native steps.
                        # Reusing an old packet would incorrectly renew its lease.
                        self.cycle(advance_physics=False)
                        ctl = self.controller
                    else:
                        self.publish(self.clock())
                ctl.step(self.clock())
        self.publish(self.clock())

    def apply_scene_action(self, action):
        """Only a simulation backend may opt into explicit scene resets."""
        self.controller.hold("scene_change_unsupported")

    def scene_state(self):
        return {}

    def publish(self, now):
        self._owned()
        ctl = self.controller
        if self.model is not None:  # CPU-only logic fixtures have no native model.
            mujoco.mj_kinematics(self.model, ctl.data)
        state = dict(owner_id=self.owner_id, sequence=self.accepted_sequence, simulation_s=float(ctl.data.time),
                     enabled=bool(ctl.enabled), stopped=bool(ctl.stopped), reason=str(ctl.reason),
                     torque_enabled=bool(ctl.torque_enabled), recovery_held=ctl.recovery_held(now), recovery_active=ctl.recovery_valid(now), contacts=ctl.collisions(),
                     submit_status=dict(ctl.submit_status),
                     faults=list(ctl.faults), tools={s: pose_json(ctl.tool(s)) for s in self.description["arms"]},
                     goals={s: pose_json(ctl.goals[s]) if s in ctl.goals else None for s in self.description["arms"]},
                     qpos=ctl.data.qpos.tolist(), qvel=ctl.data.qvel.tolist(), effort=ctl.data.actuator_force.tolist(), reference=ctl.reference.tolist(),
                     grabs={s: s in ctl.mapper.grabs for s in self.description["arms"]}, gain=float(ctl.mapper.gain), server_ms=now/1e6)
        state.update(self.scene_state())
        snapshot = Capture(tuple(ctl.data.qpos.tolist()), now, json_bytes(state))
        with self.lock:
            self.state, self.snapshot, self.owner_acquired_ns = state, snapshot, now
        self._record("state", dict(acquired_ns=now, **state))

    def owner_loop(self):
        try:
            self.initialize_owner()
            deadline = self.clock()
            while not self.stop_event.is_set():
                self.cycle()
                deadline += PERIOD_NS*self.physics_steps_per_cycle//10
                remaining = (deadline-self.clock())/1e9
                if remaining < 0:
                    deadline = self.clock()
                else:
                    self.stop_event.wait(remaining)
        except Exception as error:
            with self.lock:
                self.owner_error = type(error).__name__ + ": " + str(error)
            if self.controller is not None:
                self.controller.faults.append("owner_failure:" + self.owner_error)
            self._record("events", dict(event="owner_failure", reason=self.owner_error, server_ns=self.clock()))
            self.stop_event.set()
        finally:
            if self.controller is not None:
                try:
                    self.controller.stop()
                    self.publish(self.clock())
                except Exception as error:
                    with self.lock:
                        self.state = dict(self.state, enabled=False, stopped=True, reason="owner_failure",
                                          faults=[self.owner_error or str(error)])

    def start(self):
        self.owner_thread = threading.Thread(target=self.owner_loop, name="so101-plant-owner", daemon=False)
        self.owner_thread.start()

    def close(self):
        self.stop_event.set()
        if self.owner_thread:
            self.owner_thread.join(timeout=3)
            if self.owner_thread.is_alive():
                raise RuntimeError("owner_failed_to_stop")

    def get_state(self):
        with self.lock:
            return json.loads(json_bytes(self.state))

    def get_config(self):
        return dict(owner_id=self.owner_id, mode="dual" if len(self.description["arms"]) == 2 else "single", gain=1.,
                    tools=self.get_state()["tools"], camera=self.camera, cameras=self.cameras,
                    acquire_radius_m=.05, reference_view=True, server_ms=self.clock()/1e6)

    def get_health(self):
        now = self.clock()
        fresh = self._fresh(now)
        with self.lock:
            alive = bool(self.owner_thread and self.owner_thread.is_alive())
            return dict(ready=bool(alive and self.render_alive and fresh and not self.owner_error),
                        owner_alive=alive, render_alive=bool(self.render_alive),
                        owner_age_ms=None if self.owner_acquired_ns is None else (now-self.owner_acquired_ns)/1e6,
                        frame_age_ms=None if self.frame_acquired_ns is None else (now-self.frame_acquired_ns)/1e6,
                        frame_sequence=self.frame_sequence, owner_error=self.owner_error, render_error=self.render_error,
                        scope="synthetic MuJoCo; fixed mono camera; local functional preview only")

    def measured_snapshot(self):
        with self.lock:
            return self.snapshot

    def cache_frames(self, frames, capture, robot_overlay=None):
        if set(frames) != set(self.cameras) or any(not isinstance(v, bytes) or not v for v in frames.values()):
            raise ValueError("one_native_jpeg_per_configured_camera_required")
        with self.lock:
            if self.frame_acquired_ns is not None and capture.acquired_ns <= self.frame_acquired_ns:
                return False
            self.frames = dict(frames)
            self.frame_acquired_ns = capture.acquired_ns
            self.frame_state_json = capture.state_json
            self.robot_overlay = robot_overlay
            self.frame_sequence += 1
            self.render_alive = True
        return True

    def get_frame(self, camera):
        with self.lock:
            if camera not in self.frames:
                raise KeyError(camera)
            return self.frames[camera], self.frame_sequence, self.frame_acquired_ns

    def get_views(self):
        with self.lock:
            if self.frame_state_json is None:
                raise KeyError("native_camera_views_unavailable")
            sequence, acquired, state_json, frames, overlay = self.frame_sequence, self.frame_acquired_ns, self.frame_state_json, self.frames, self.robot_overlay
        result = dict(owner_id=self.owner_id, frame_sequence=sequence, acquired_ms=acquired/1e6, state=json.loads(state_json),
                      frames={s: "data:image/jpeg;base64," + base64.b64encode(v).decode("ascii") for s, v in frames.items()})
        if overlay is not None:
            result["robot_overlay"] = "data:image/png;base64," + base64.b64encode(overlay).decode("ascii")
        return result

    def fail_render(self, error):
        with self.lock:
            self.render_error = type(error).__name__ + ": " + str(error)
            self.render_alive = False
            self.pending_hold = "renderer_failure"
            self.pending_lifecycle.clear()
        self._record("events", dict(event="renderer_failure", reason=self.render_error, server_ns=self.clock()))


def robot_geom_ids(model):
    """Native geometry descendants of both fixed arm bases, including cameras."""
    roots = {model.body(side+"_base").id for side in SIDES
             if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side+"_base") >= 0}
    bodies = set(roots)
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return np.array([g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies], dtype=int)


def robot_rgba(rgb, segmentation, geom_ids):
    mask = (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[:, :, 0], geom_ids)
    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = mask.astype(np.uint8)*255
    rgba[~mask, :3] = 0
    return rgba


class NativeFrames:
    """Main-thread-only renderer with separate observed-state MjData."""
    def __init__(self, model, cameras, width=960, height=540, *, robot_overlay=False):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("renderer_main_thread_required")
        self.model, self.cameras = model, cameras
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.scene_option = mujoco.MjvOption()
        self.scene_option.sitegroup[:] = 0  # RGB views contain no simulator debug spheres.
        self.ident = threading.get_ident()
        self.overlay_enabled = robot_overlay
        self.overlay_png = None
        self.robot_geoms = robot_geom_ids(model) if robot_overlay else None

    def render(self, snapshot):
        if threading.get_ident() != self.ident:
            raise RuntimeError("renderer_owner_thread_required")
        self.data.qpos[:] = snapshot.qpos
        mujoco.mj_forward(self.model, self.data)
        result = {}
        self.overlay_png = None
        main_rgb = None
        for alias, name in self.cameras.items():
            self.renderer.update_scene(self.data, camera=name, scene_option=self.scene_option)
            stream = io.BytesIO()
            rgb = self.renderer.render().copy()
            Image.fromarray(rgb).save(stream, format="JPEG", quality=90)
            result[alias] = stream.getvalue()
            if alias == "main":
                main_rgb = rgb
        if self.overlay_enabled:
            self.renderer.update_scene(self.data, camera=self.cameras["main"], scene_option=self.scene_option)
            self.renderer.enable_segmentation_rendering()
            try:
                segmentation = self.renderer.render().copy()
            finally:
                self.renderer.disable_segmentation_rendering()
            overlay = io.BytesIO()
            Image.fromarray(robot_rgba(main_rgb, segmentation, self.robot_geoms)).save(overlay, format="PNG")
            self.overlay_png = overlay.getvalue()
        return result, snapshot

    def close(self):
        self.renderer.close()


def make_handler(runtime, web_root):
    root = Path(web_root).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, code, payload, mime="application/json", headers=None):
            body = json_bytes(payload) if mime == "application/json" else payload
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/api/scene" and hasattr(runtime, 'get_visual_scene'):
                return self.respond(200, runtime.get_visual_scene(), 'application/json; charset=utf-8', {'Content-Encoding':'gzip'})
            if url.path == "/api/state":
                return self.respond(200, runtime.get_state())
            if url.path == "/api/config":
                return self.respond(200, runtime.get_config())
            if url.path == "/api/health":
                return self.respond(200, runtime.get_health())
            if url.path == "/api/phone":
                return self.respond(200, runtime.phone.get())
            if url.path == "/api/views":
                try:
                    return self.respond(200, runtime.get_views())
                except KeyError:
                    return self.respond(503, dict(reason="native_camera_views_unavailable"))
            if url.path == "/api/frame.jpg":
                try:
                    camera = parse_qs(url.query).get("camera", ["main"])[0]
                    frame, sequence, acquired = runtime.get_frame(camera)
                    return self.respond(200, frame, "image/jpeg", {"X-Frame-Sequence": sequence, "X-Frame-Acquired-Ms": acquired/1e6})
                except KeyError:
                    return self.respond(503, dict(reason="native_camera_frame_unavailable"))
            name = unquote(url.path).lstrip("/") or "index.html"
            allowed = {"index.html", "app.js", "style.css", "preview.mjs", "immersive.mjs", "hybrid.mjs", "camera.html", "camera.mjs", "camera-helper.mjs", "vendor/three.module.js", "vendor/three.core.js"}
            path = (root / name).resolve()
            if name not in allowed or not path.is_relative_to(root) or not path.is_file():
                return self.respond(404, dict(reason="not_found"))
            mime = "text/html" if name.endswith(".html") else "text/css" if name.endswith(".css") else "text/javascript"
            return self.respond(200, path.read_bytes(), mime)

        def do_POST(self):
            if self.path not in ("/api/session", "/api/input", "/api/phone/session", "/api/phone/frame", "/api/phone/stop"):
                return self.respond(404, dict(reason="not_found"))
            original_text = ""
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body_limit = PHONE_BODY_BYTES if self.path == "/api/phone/frame" else 65536
                if not 0 < length <= body_limit:
                    raise ValueError("bounded_json_body_required")
                self.connection.settimeout(2.)
                original_text = self.rfile.read(length).decode("utf-8")
                raw = json.loads(original_text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("finite_json_required")))
                if not isinstance(raw, dict):
                    raise ValueError("json_object_required")
                json_bytes(raw)
                if self.path == "/api/phone/session":
                    result = runtime.phone.create_session(raw.get("client_ms"))
                elif self.path == "/api/phone/frame":
                    result = runtime.phone.receive_frame(raw)
                elif self.path == "/api/phone/stop":
                    result = runtime.phone.stop(raw.get("session"))
                else:
                    result = runtime.create_session(raw.get("client_ms")) if self.path == "/api/session" else runtime.receive_input(raw)
                return self.respond(200, result)
            except (ValueError, TypeError, TimeoutError) as error:
                if self.path == "/api/input":
                    runtime.reject_malformed_input(str(error), original_text)
                return self.respond(400, dict(accepted=False, reason=str(error)))

    return Handler

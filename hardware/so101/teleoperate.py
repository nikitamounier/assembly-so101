from contextlib import ExitStack
import json
import logging
from pathlib import Path
from shutil import which
from signal import SIGHUP, SIGINT, SIGTERM, signal
from subprocess import Popen, run
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from calibrate import CALIBRATION_PATH, load_bundle

pairs = load_bundle()['pairs']


def stop_on_signal(_signal, _frame):
    raise KeyboardInterrupt


signal(SIGHUP, stop_on_signal)
signal(SIGTERM, stop_on_signal)


def check_ports(*pair_names):
    uncalibrated = [name for name in pair_names if name not in pairs]
    if uncalibrated:
        raise SystemExit(f'Calibrate {", ".join(uncalibrated)} with calibrate.py or supply {CALIBRATION_PATH.name}.')
    ports = [pairs[name][role]['port'] for name in pair_names for role in ('leader', 'follower')]
    if len({Path(port).resolve() for port in ports}) != len(ports):
        raise SystemExit('Each arm must have a distinct serial port in the calibration bundle.')
    missing = [port for port in ports if not Path(port).exists()]
    if missing:
        raise SystemExit(f'Connect the missing arm USB ports: {", ".join(missing)}')
    if lsof := which('lsof'):
        occupied = run([lsof, '-t', *ports], capture_output=True, text=True)
        if occupied.stdout.strip():
            raise SystemExit('A selected pair is already open. Stop it in its Terminal window and try again.')


def launch_both():
    check_ports('front', 'rear')
    processes = []
    try:
        for selection in ('--front-wowrobo', '--rear'):
            command = [sys.executable, '-u', str(Path(__file__).resolve()), selection]
            processes.append(Popen(command, start_new_session=True))
        while True:
            for process in processes:
                if (status := process.poll()) is not None:
                    return status
            sleep(0.1)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(SIGINT)
        for process in processes:
            process.wait()


args = sys.argv[1:]
if args == ['--both']:
    raise SystemExit(launch_both())
if args not in ([], ['--rear'], ['--front-wowrobo']):
    raise SystemExit('Choose --both, --rear, or --front-wowrobo.')
front_pair = args == ['--front-wowrobo']
pair_name = 'front' if front_pair else 'rear'
check_ports(pair_name)
pair = pairs[pair_name]
logging.basicConfig(
    filename=Path(__file__).with_name('teleoperate-front-wowrobo.log' if front_pair else 'teleoperate.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    force=True,
)
logging.getLogger('lerobot.robots.utils').setLevel(logging.ERROR)

# LeRobot loads one file per device at construction. Materialize those files only
# for the constructor so the saved bundle is the single persistent input.
with TemporaryDirectory(prefix='so101-calibration-') as temp:
    calibration_dir = Path(temp)
    for role in ('leader', 'follower'):
        arm = pair[role]
        (calibration_dir / f"{arm['id']}.json").write_text(json.dumps(arm['joints']))
    leader = SO101Leader(SO101LeaderConfig(
        port=pair['leader']['port'], id=pair['leader']['id'], calibration_dir=calibration_dir, use_degrees=False,
    ))
    follower = SO101Follower(SO101FollowerConfig(
        port=pair['follower']['port'], id=pair['follower']['id'], calibration_dir=calibration_dir, use_degrees=False,
    ))
bus = follower.bus
fps = 60


def start_from_measured_pose(enable_torque):
    def enable(*args, **kwargs):
        position = bus.sync_read('Present_Position', normalize=False, num_retry=2)
        for joint, value in position.items():
            calibration = follower.calibration[joint]
            position[joint] = min(calibration.range_max, max(calibration.range_min, value))
        bus.sync_write('Goal_Position', position, normalize=False, num_retry=2)
        if bus.sync_read('Goal_Position', normalize=False, num_retry=2) != position:
            raise RuntimeError('Follower could not confirm the starting hold target')
        enable_torque(*args, **kwargs)
    return enable


def front_target(action):
    # Both arms were homed to the same encoder midpoint. Preserve that reference
    # even when the sweeps recorded different endpoints around it.
    target = action.copy()
    for joint, source in leader.calibration.items():
        if joint == 'gripper':
            continue
        destination = follower.calibration[joint]
        key = f'{joint}.pos'
        ticks = source.range_min + (action[key] + 100) / 200 * (source.range_max - source.range_min)
        value = 200 * (ticks - destination.range_min) / (destination.range_max - destination.range_min) - 100
        target[key] = min(100, max(-100, value))
    return target


try:
    with ExitStack() as cleanup:
        for device in (leader, follower):
            device.bus.connect()
            cleanup.callback(device.bus.disconnect, disable_torque=True)
            if not device.is_calibrated:
                raise RuntimeError(f'Calibration mismatch for {device.id}')

        follower_start = follower.get_observation()
        if front_pair:
            leader_start = front_target(leader.get_action())
            for joint, calibration in follower.calibration.items():
                if joint == 'gripper':
                    continue
                key = f'{joint}.pos'
                degrees_per_point = (calibration.range_max - calibration.range_min) * 360 / (4095 * 200)
                mismatch = abs(leader_start[key] - follower_start[key]) * degrees_per_point
                if mismatch > 20:
                    raise RuntimeError(f'Match the two starting poses before launching ({joint}: {mismatch:.0f}° apart)')

        leader.configure()
        with patch.object(bus, 'enable_torque', start_from_measured_pose(bus.enable_torque)):
            follower.configure()

        print(f'{pair_name.capitalize()} SO-101 pair live. Move the leader. Ctrl+C to stop.')
        logging.info('SO-101 teleoperation live at %d Hz with calibrated range mapping', fps)
        alignment_started = monotonic()
        next_log = 0.0
        while True:
            started = monotonic()
            action = leader.get_action()
            target = front_target(action) if front_pair else action.copy()
            if closed := pair.get('gripper_closed'):
                gripper_opening = max(0.0, action['gripper.pos'] - closed['leader'])
                target['gripper.pos'] = min(100.0, closed['follower'] + gripper_opening)
            if not front_pair:
                # After a reconnect the rear pair can settle at different angles.
                blend = min(1.0, (started - alignment_started) / 1.5)
                if blend < 1.0:
                    target = {j: follower_start[j] + blend * (value - follower_start[j]) for j, value in target.items()}
            follower.send_action(target)
            if started >= next_log:
                measured = follower.get_observation()
                logging.info(
                    'leader %s follower %s gripper target %.1f',
                    {j: round(v, 1) for j, v in action.items()},
                    {j: round(v, 1) for j, v in measured.items()},
                    target['gripper.pos'],
                )
                next_log = started + 2.0
            sleep(max(0, 1 / fps - (monotonic() - started)))
except KeyboardInterrupt:
    print(f'{pair_name.capitalize()} teleoperation stopped. Torque released.')
    logging.info('Stopped by operator')
except Exception:
    logging.exception('Teleoperation stopped with an error')
    raise

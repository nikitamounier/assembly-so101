import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


CALIBRATION_PATH = Path(__file__).resolve().with_name('so101-calibration.json')


def load_bundle():
    if not CALIBRATION_PATH.is_file():
        return {'version': 1, 'pairs': {}}
    bundle = json.loads(CALIBRATION_PATH.read_text())
    if bundle.get('version') != 1:
        raise ValueError('Unknown SO-101 calibration file version.')
    return bundle


def save_bundle(bundle):
    with NamedTemporaryFile(
        mode='w', dir=CALIBRATION_PATH.parent, prefix=f'{CALIBRATION_PATH.name}.',
        suffix='.tmp', delete=False,
    ) as output:
        temporary_path = Path(output.name)
        try:
            json.dump(bundle, output, indent=2)
            output.write('\n')
            output.close()
            temporary_path.replace(CALIBRATION_PATH)
        finally:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Calibrate one SO-101 pair and save its local bundle.')
    parser.add_argument('pair', choices=('front', 'rear'))
    parser.add_argument('--leader-port', required=True)
    parser.add_argument('--follower-port', required=True)
    parser.add_argument('--capture-gripper-closure', action='store_true')
    args = parser.parse_args()
    if Path(args.leader_port).resolve() == Path(args.follower_port).resolve():
        parser.error('Use a different USB port for each arm.')

    bundle = load_bundle()
    result = {}
    closed = {}
    with TemporaryDirectory(prefix='so101-calibration-') as temp, ExitStack() as cleanup:
        devices = {
            'follower': SO101Follower(SO101FollowerConfig(
                port=args.follower_port, id=f'{args.pair}_follower', calibration_dir=Path(temp),
                use_degrees=False,
            )),
            'leader': SO101Leader(SO101LeaderConfig(
                port=args.leader_port, id=f'{args.pair}_leader', calibration_dir=Path(temp),
                use_degrees=False,
            )),
        }
        originals = {}
        for role, device in devices.items():
            bus = device.bus
            bus.connect()
            cleanup.callback(bus.disconnect, disable_torque=False)
            if any(bus.sync_read('Torque_Enable', normalize=False).values()):
                raise RuntimeError(f'{role.capitalize()} torque is active. Stop teleoperation first.')
            originals[role] = bus.read_calibration()

        touched = []
        try:
            for role, device in devices.items():
                print(f'\n{role.capitalize()} on {device.config.port}. Support the arm and follow the torque-off prompts.')
                touched.append(role)
                device.calibrate()
                if not device.is_calibrated:
                    raise RuntimeError(f'{role.capitalize()} calibration readback differs.')
                result[role] = {
                    'id': device.id,
                    'port': device.config.port,
                    'joints': {name: asdict(cal) for name, cal in device.calibration.items()},
                }
                if args.capture_gripper_closure:
                    input(f'Close the {role} gripper/trigger by hand and press Enter while holding it: ')
                    closed[role] = float(device.bus.read('Present_Position', 'gripper', num_retry=2))
            if closed:
                result['gripper_closed'] = closed
            bundle['pairs'][args.pair] = result
            save_bundle(bundle)
        except BaseException as error:
            for role in reversed(touched):
                bus = devices[role].bus
                try:
                    bus.disable_torque(num_retry=2)
                    bus.write_calibration(originals[role])
                    if bus.read_calibration() != originals[role]:
                        raise RuntimeError('restored registers differ from the original')
                except Exception as restore_error:
                    error.add_note(f'Could not restore the {role} calibration: {restore_error}')
            raise
    print(f'Saved and verified {args.pair} calibration in {CALIBRATION_PATH}')


if __name__ == '__main__':
    main()

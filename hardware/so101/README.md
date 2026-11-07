# Physical SO-101 teleoperation

Goal: calibrate and run one or two physical SO-101 leader/follower pairs with a single local calibration bundle.

## Set up

Use Python 3.12 on macOS or Linux with the Feetech serial arms. Clamp both bases and connect USB and the power supplies specified for your kit. Use an isolated environment for the hardware tools.

```sh
cd hardware/so101
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/lerobot-find-port
```

Identify each port by disconnecting the corresponding USB cable when prompted, then reconnect it. On macOS the path is typically `/dev/tty.usbmodem…` or its `/dev/cu.usbmodem…` counterpart. On Linux it is typically `/dev/ttyACM0` or `/dev/ttyUSB0`.

## Calibrate

Select `front` for matching arms homed to the same pose. It mirrors joint angles within the follower's measured travel. Select `rear` for the calibrated-range mapping used with the mixed-vendor pair. A single-pair setup only needs one of the two.

```sh
.venv/bin/python calibrate.py front \
  --leader-port /dev/ttyACM0 --follower-port /dev/ttyACM1
```

Follow LeRobot's prompts to set each midpoint and move each joint through its travel by hand with torque off. For a pair whose closed trigger and jaw readings differ, add `--capture-gripper-closure` and hold each control closed when asked. The tool verifies the saved ranges against the servos and saves the result in `so101-calibration.json`. Calibrating the second pair adds it to the same file. [LeRobot SO-101 calibration](https://huggingface.co/docs/lerobot/so101#calibrate)

An existing version-1 `so101-calibration.json` can be copied beside the launcher for the same physical arms. Update the local USB paths if they changed. Keep this machine-specific bundle with the hardware setup.

## Run

Start each selected pair from similar, supported poses with the follower's sweep clear.

```sh
./teleoperate.command
```

Choose **1** for both pairs, **2** for front, or **3** for rear. On macOS you can also double-click the launcher. Press **Ctrl+C** in that terminal to stop the selected pairs and release torque. After a USB disconnect, reconnect the cable and launch the pair again.

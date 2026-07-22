# MD gripper controller

This program provides three commands for the motor at CAN ID 21:

- `home`: move into the mechanical stop with constant torque and zero the encoder
- `position`: move to a position in radians with a trapezoidal profile
- `close`: apply and continuously hold constant torque until Ctrl+C

## Important: torque is not current

`setTargetTorque()` and the command-line values use **Nm**, not amperes. The MD
controller converts requested torque to motor current using the torque constant in
its motor configuration. `setCurrentLimit()` only changes the safety limit; it
does not command a current.

The sign depends on the installation. The current gripper uses a negative value
to close. Begin with a low magnitude and verify the direction with the mechanism
clear.

## Build

The installed CANdle SDK requires C++20. From this directory run:

```bash
./build.sh
```

The executable is `build/gripper_control`.

## Commands

```bash
# Home with the default -0.6 Nm
./build/gripper_control home

# Home with an explicit torque
./build/gripper_control home -0.3

# Move to 2 radians (optional velocity and acceleration follow)
./build/gripper_control position 2.0 20 40

# Apply and hold -0.2 Nm; press Ctrl+C to release and disable the motor
./build/gripper_control close -0.2
```

`close` keeps the program running, the drive enabled, and `RAW_TORQUE` active.
It refreshes the fixed torque target at 50 Hz and prints target torque, measured
torque, and position twice per second. This supports drive configurations with a
command watchdog and provides useful fault diagnostics.
Pressing Ctrl+C calls `Gripper::stop()` and disables the drive. An application
that embeds this library must likewise keep running and call `Gripper::stop()`
on its shutdown or emergency-stop path.

## Why the previous torque test did not work

The old Python `main.py` contained this expression:

```python
pc.MotionMode_t.RAW_TORQUE
```

It names the enum but does not send anything to the controller. Consequently the
motor remained in position mode when `setTargetTorque()` was called. The fixed
code calls `gripper.close()`, which first calls `setMotionMode(RAW_TORQUE)` and
then sends the torque target.

The position command now selects `POSITION_PROFILE`; profile velocity and
acceleration do not apply in plain `POSITION_PID` mode.

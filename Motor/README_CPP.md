# MD gripper controller and Vision System bridge

Motor communication is implemented only in C++ using the CANdle SDK. There is
no Python CANdle motor backend. Two executables are built:

- `gripper_control`: standalone diagnostic CLI for CAN ID 21
- `gripper_bridge`: persistent, safety-checked command/feedback process used by
  the Vision System

The standalone program provides staged diagnostic commands:

- `probe`: connect, apply limits, and read feedback while remaining disabled
- `enable-test`: briefly enable in IDLE without a motion command, then disable
- `clear-faults`: clear controller faults while disabled
- `home`: move into the mechanical stop with constant torque and zero the encoder
- `position`: move to a position in radians with a trapezoidal profile
- `torque-test`: ramp torque for a bounded duration, then automatically release
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

The executables are `build/gripper_control` and `build/gripper_bridge`.

## Commands

```bash
# Safest first test: connection and feedback, drive remains disabled
./build/gripper_control probe

# Enable in IDLE without a motion target, then immediately stop and disable
./build/gripper_control enable-test

# Clear faults without enabling
./build/gripper_control clear-faults

# Home with the default -0.6 Nm
./build/gripper_control home

# Home with an explicit torque
./build/gripper_control home -0.3

# Move to 2 radians; this reduced-rate example is preferable for initial tests
./build/gripper_control position 2.0 0.5 2.0

# Ramp to -0.1 Nm for 2 seconds, then automatically release and disable
./build/gripper_control torque-test -0.1 2

# Apply and hold -0.2 Nm; press Ctrl+C to release and disable the motor
./build/gripper_control close -0.2
```

Run these only while the Vision System is not connected to the adapter. Every
command applies the `0.5 A` current limit and `10 Nm` controller limit first,
and every exit path attempts zero torque, IDLE, disable, and disconnect.

`torque-test` and `close` ramp at `0.5 Nm/s`, refresh the target at 50 Hz, and
print position, velocity, measured torque, and temperature. `torque-test` is
limited to 10 seconds and is the preferred first torque check. `close` keeps the
program running with raw-torque mode active until Ctrl+C.
Pressing Ctrl+C calls `Gripper::stop()` and disables the drive. An application
that embeds this library must likewise keep running and call `Gripper::stop()`
on its shutdown or emergency-stop path.

## Vision System bridge

Do not launch `gripper_bridge` manually during normal operation. The Vision
System hardware backend owns one bridge process for the complete connected
session. It initializes the motor with the drive disabled and rejects enable or
motion until the validated safety limits have been supplied and applied.

The bridge accepts granular commands for enable/disable, fault clearing,
homing support, profiled position, raw torque, torque refresh, feedback, and
shutdown. Limits for current, controller torque, closing torque/direction,
position, velocity, acceleration, and temperature are checked again in C++.
Any failed or malformed command triggers a best-effort zero-torque, IDLE, and
disable sequence before an error is returned.

The position command selects `POSITION_PROFILE`; profile velocity and
acceleration do not apply in plain position PID mode. Torque commands explicitly
select raw-torque mode before setting their first target.

# Vision System

This program detects red markers on a Fin Ray gripper, assigns permanent marker
IDs, tracks their deformation relative to a real moving origin marker, tracks
one user-selected blue target relative to that origin, records force-calibration
datasets, trains a force model, and performs live force prediction.

The PySide6 application also integrates the existing CANdle/MD gripper project
through a guarded motor subsystem. It starts disconnected and uses the mock
backend by default. No motor is enabled, homed, or moved at GUI startup.

## Integrated architecture

```text
MainWindow (GUI thread)
├── ApplicationController
│   └── CameraWorker / ApplicationState (camera worker QThread)
├── MotorController
│   └── MotorWorker / MotorBackend (motor worker QThread)
├── ExperimentController (detached snapshots and coordination only)
└── TrainingController (cancellable audit/training child process)
```

Only `MotorWorker` owns or calls a motor backend. Only `CameraWorker` owns the
mutable `ApplicationState`. The experiment coordinator observes copied scalar
vision and motor snapshots; motor objects are never stored in vision state.
Dataset validation still goes through `DatasetSession.add_sample()`. Training
reuses `train_force_model.py` in a separate process.

## Marker layout and configuration

The system uses three marker roles:

- `REFERENCE_ORIGIN`: exactly one real red marker attached to the moving finger.
  Select it manually during reference setup. It provides translation
  compensation and is not part of either deforming line.
- `LINE_A_*` and `LINE_B_*`: deforming markers used for force features.
- `RIGID_REFERENCE_*`: optional markers used only for translation-and-rotation
  compensation.

Each detected marker can have only one role and can belong to only one line. A
marker at the visual crossing of the two lines must therefore be assigned to
either line A or line B; it cannot be counted in both.

```yaml
reference_ransac:
  expected_points_line_a: 8
  expected_points_line_b: 8

tracking:
  expected_deforming_point_count: 16
  expected_reference_origin_point_count: 1
```

The required rule is:

```text
expected_deforming_point_count
    = expected_points_line_a + expected_points_line_b
```

If the stated total of 15 includes the origin marker, there are only 14
deforming markers, so the two line counts must add up to 14 instead.

The current `camera-config.yaml` contains `8 + 8 = 16` deforming markers. Change
those values to match the physical marker layout before creating a new
reference profile. Restart the program after editing the YAML file.

## Starting the program

### PySide6 graphical interface

Install the GUI and test dependencies in the project environment once:

```bash
cd "/home/janek/Desktop/Programming/Programming/Vision System"
../../virtual-env/bin/pip install -r requirements-gui.txt
```

Start the graphical application:

```bash
../../virtual-env/bin/python gui_main.py
```

Use `--no-camera` to inspect the interface or run it on a computer without a
connected RealSense camera:

```bash
../../virtual-env/bin/python gui_main.py --no-camera
```

The window contains Camera, Tuning, Dataset, Motor, Collection, and Training
tabs. The Camera workspace keeps
the live image visible on the left while Runtime, red Calibration, Blue blob,
and Reference setup controls are selected on the right. This lets origin selection and
reacquisition happen without switching away from the reference workflow. The
command console at the bottom accepts every command in the existing `COMMANDS`
registry. Buttons and checkboxes use that same command layer. Camera acquisition
and image processing run on one dedicated Qt worker thread. Motor communication
runs on a second worker thread, and model training runs in a cancellable child
process. Closing the window first stops/disables/disconnects the motor, then
stops the camera worker; a running training process is also terminated.

The Tuning tab distinguishes editor values, values applied to the running
process, and values saved in `camera-config.yaml`. Apply validates editor values;
Save Configuration writes the applied values atomically. Camera-setting changes
restart the camera. Resolution and structural reference changes require clearing
an accepted profile first.

## Motor backend and safe operation

The dedicated configuration is `motor-config.yaml`. Its default is:

```yaml
motor:
  backend: mock
```

Keep `backend: mock` for development, GUI checks, and the default test suite.
Changing it to `hardware` only selects the real adapter; the drive still starts
disconnected and requires an explicit **Connect** action.

The hardware path uses only the C++ CANdle SDK. The Vision System starts the
persistent `Motor/build/gripper_bridge` executable and exchanges a small
line-oriented command/feedback protocol with it. Python does not import or load
a CANdle binding, and all MD SDK calls run inside the C++ process. Build the bridge
before selecting the hardware backend:

```bash
cd "/home/janek/Desktop/Programming/Programming/Motor"
./build.sh
```

The C++ bridge starts with the drive disabled, independently enforces the
configured safety limits, and commands zero torque, IDLE, and disable when a
command fails or the bridge shuts down.

Confirmed limits and units are:

| Setting | Value |
| --- | ---: |
| Closing direction | negative |
| Homing torque | `-0.6 Nm` |
| Application closing magnitude | at most `4 Nm` |
| Controller torque magnitude | at most `10 Nm` |
| Current limit | `0.5 A` |
| Position range | configurable within `-1.0..2.2 rad`, including `0 rad` |
| Open position | `2.0 rad` |
| Position tolerance | `0.5 rad` |
| Maximum profile velocity | `5 rad/s` |
| Maximum profile acceleration | `20 rad/s²` |
| Maximum temperature | `50 °C` |
| Torque ramp | `0.5 Nm/s` |
| Measured-torque tolerance | `±0.1 Nm` |
| Lower-torque backoff | `0.3 rad` toward open |

The controller maximum is configured as the positive magnitude `10 Nm`; the
application rejects closing commands outside `-4..0 Nm` and rejects positive
closing torque. Position, profile, temperature, feedback, timeout, and homing
requirements are checked centrally.

> **Software stop is not a physical emergency stop.** Install and keep an
> accessible physical emergency-stop for personnel and equipment safety. A Qt
> event, operating-system process, USB link, or computer can fail.

Normal hardware procedure after the physical safety system has been checked:

1. Select the Motor tab and verify that the backend says `hardware` only when a
   hardware session is intended.
2. Click **Connect**. Connection applies the `0.5 A` current limit and `10 Nm`
   controller limit while leaving the drive disabled.
3. Clear faults only while disabled, then click **Enable drive**.
4. Confirm **Home**. Homing is incremental and remains interruptible by the
   direct red **STOP MOTOR** button.
5. Use **Open gripper** or a validated position command.
6. Select raw motor torque, or calculated finger force after entering the real
   pulley radius and efficiency. The conversion is:

   ```text
   finger_force = (motor_torque / pulley_radius) * efficiency
   motor_torque = finger_force * pulley_radius / efficiency
   ```

   Radius and efficiency have no guessed defaults. Force uses the signed motor
   convention, so closing force is negative with the confirmed direction.
   Torque can be adjusted while
   holding. Reducing its magnitude releases torque, opens `0.3 rad`, and
   reapproaches with the new target to reduce static-friction bias.
7. **Release torque** and **STOP MOTOR** both command zero torque, IDLE, and
   disable under the confirmed policy. Disconnect when finished.

The **Configured motor ranges** box is editable only while the motor is
disconnected. It sets both position endpoints (`-1..0 rad` for the selectable
minimum and up to `2.2 rad` for the maximum), the Open-button target, and the
negative closing-torque limit (never beyond `-4 Nm`; the other endpoint is
fixed at `0 Nm`). The position range must include the homed `0 rad` origin.
**Apply and save ranges** validates the selection, updates the command fields,
and atomically saves `motor-config.yaml`. The new limits are sent to the C++
bridge on the next connection. The controller-register ceiling remains fixed
at a magnitude of `10 Nm`.

Motor feedback shows state, position, velocity, target and measured torque,
torque error and confirmation, feedback age, temperature, elapsed motion time,
warnings, and errors. Enable **Show detailed motor debug trace** to see the GUI
request, the exact command sent to the C++ bridge (`TX`), its response (`RX`),
native CANdle/SDK output (`NATIVE`), interpreted feedback samples, state
transitions, and confirmed or failed outcomes. The trace also reports the GUI
user/group IDs and bridge-process PID, which helps diagnose USB permission
differences between a terminal launch and a desktop GUI launch. Use **Copy
trace** to include the complete trace in a bug report.

Feedback, homing, motion completion, watchdog refresh, and stop handling use an
approximately 20 ms worker timer; every hardware operation is forwarded to the
persistent C++ bridge, so Qt buttons never call a Python motor binding. Starting
a torque ramp is only reported as accepted; success is not reported until the
command reaches the target and measured torque is within the configured
`±0.1 Nm` tolerance.

### Terminal/OpenCV interface

Connect the RealSense camera, open a terminal, and run:

```bash
cd "/home/janek/Desktop/Programming/Programming/Vision System"
source ../../virtual-env/bin/activate
python main.py
```

To test only the camera:

```bash
python camera_check.py
```

The terminal continuously shows a `Command>` prompt. Type commands directly and
press Enter; pressing `t` first is no longer necessary. Keep the terminal
focused when typing commands. Keyboard shortcuts apply when the OpenCV camera
window is focused.

Run this at any time to list the text commands:

```text
help
```

## First-time reference setup

The gripper must be unloaded and all required red markers must be clearly
visible.

1. Check the camera image. Every valid red blob should have a detection label
   such as `D0`, `D1`, and so on.
2. Start reference setup:

   ```text
   reference_start
   ```

3. Arm origin selection:

   ```text
   reference_origin_select
   ```

4. Left-click the physical origin marker in the camera window. It should be
   labelled `ORIGIN`.
5. Fit the two deforming lines:

   ```text
   reference_preview
   ```

6. Check that the terminal reports `Reference valid=True`. Inspect the
   `LINE_A_00`, `LINE_A_01`, ... and `LINE_B_00`, `LINE_B_01`, ... labels in the
   image.
7. Accept and save the reference:

   ```text
   reference_accept
   reference_save
   ```

The default saved profile is `reference-profile.json`.

If the preview is wrong, run `reference_reject`, correct the scene or
configuration, and preview again. Use `reference_origin_clear` to select a
different origin, or `reference_clear` to discard the entire setup.

## How line A and line B are chosen

Line names are assigned deterministically from their positions in the unloaded
reference image; they are not chosen by the order in which RANSAC happens to
find them. With the default `direction_rule: base_to_tip`, marker order along a
line normally progresses from left to right in the camera image.

During `reference_preview`:

- line A is drawn light blue/cyan;
- line B is drawn orange;
- the permanent marker labels beginning with `LINE_A_` or `LINE_B_` are the
  definitive identification.

`show_lines on` controls the fitted-reference proposal overlay. It will not
create a line before `reference_preview`. When a saved profile is loaded in a
new run, no RANSAC proposal exists, so the permanent point labels are the useful
line identification. Use `show_geometry on` to display configured runtime
connections.

Display commands include:

```text
show_points on
show_lines on
show_outliers on
show_geometry on
show_features on
show_status on
show_warnings on
show_circles on
show_mask_only off
```

## Loading a saved reference

On later runs, load the profile and start tracking:

```text
reference_load
tracking_start
tracking_status
```

If a different path was used:

```text
reference_load path/to/reference-profile.json
```

The loaded profile must match the line counts and configuration versions in
`camera-config.yaml`.

If the origin becomes invalid, data recording and force prediction stop using
that frame. When the program requests manual recovery, enter:

```text
origin_reacquire
```

Then click the same physical origin marker in the camera window.

## Recording force-calibration data

An accepted reference and active tracking are required. Keep the force stable
while acquiring each batch of frames.

Example experiment:

```text
dataset_start finger_calibration 0
dataset_sample 100
dataset_force 5
dataset_sample 100
dataset_force 10
dataset_sample 100
dataset_force 15
dataset_sample 100
dataset_status
dataset_stop
```

Meaning:

- `dataset_start EXPERIMENT_ID KNOWN_FORCE_N` creates a new, non-overwriting
  acquisition folder.
- `dataset_force KNOWN_FORCE_N` changes the force label for subsequent samples.
- `dataset_sample FRAME_COUNT` queues that many valid frames for recording.
- `dataset_abort` discards the current pending batch without changing accepted
  rows.
- `dataset_status` shows accepted, rejected, and pending counts.
- `dataset_stop` closes the active acquisition session.

The force label and a unique step ID are frozen when a batch is queued. The GUI
disables force changes while frames are pending, and the command layer rejects
the same race. Invalid frames are rejected and retried according to
`camera-config.yaml`. Manual batch spacing uses `dataset.sample_interval_ms`;
protocol collection uses its own explicit interval.

Data is written under:

```text
datasets/<experiment-id>/<acquisition-batch-id>/
├── metadata.yaml
├── samples.csv
├── rejections.csv          # present if attempts were rejected
└── images/                 # when save_images is enabled
```

The CSV contains the immutable known force, step direction and repetition, raw
and origin-relative coordinates, point statuses, geometry quality,
deterministic feature names, and feature values. Automated samples additionally
record the closest valid permanent deforming red point to the current valid blue
target and detached motor telemetry with its monotonic age. Motor telemetry is
auxiliary metadata and is never copied into `known_force_N`.

For a useful calibration dataset:

- include an unloaded `0 N` condition;
- record multiple known force levels over the intended operating range;
- collect many stable frames at every force;
- avoid moving or hiding the origin marker;
- use the same accepted reference profile throughout an experiment.

### Operator-guided Collection tab

`collection-protocol.yaml` is a safe template. It supports generated force
levels or explicit steps, repetitions, loading/unloading sequence, samples per
step, settling time, sample interval, consecutive stability duration, quality
thresholds, motor telemetry age, and the confirmation policy. It contains only
a `0 N` placeholder; replace force levels with the actual experiment plan.

The force source is currently `OperatorForceSource`: there is no load cell. The
operator must adjust and confirm the known force observed with the special
finger. Motor torque in Nm must never be entered as force in N.

Workflow:

1. Calibrate red markers and the one blue target; accept the reference and start
   valid tracking.
2. Connect, enable, and home the motor if the protocol uses it.
3. Load the protocol in Collection. Preflight checks camera state, accepted
   reference, exact marker count, tracking/origin/geometry/schema, blue target,
   dataset destination, and motor readiness.
4. Set or adjust raw torque or calculated finger force in Motor. The motor ramps
   toward the target at `0.5 Nm/s`.
5. When the special finger shows the intended known force, enter that force in
   Collection and click **Confirm force and save step**.
6. Recording begins only after continuous vision/motor stability and settling.
   Accepted rows save all current origin-relative point positions. Sampling is
   interval-controlled and rejects stale motor telemetry.
7. Review, accept, retry, or skip the step. Abort preserves already accepted
   rows. Atomic `collection-progress.yaml` metadata records progress.
8. For a consecutive repeated step at the same force, the coordinator can stop,
   re-enable, open to `2.0 rad`, and reapply the previously captured torque. If
   **Require confirmation before each transition** is enabled, the operator must
   approve that reset first. The force still requires operator verification
   before every save.

Stability requires consecutive valid tracking, origin, geometry, blue target,
minimum quality, low feature movement, low motor velocity, torque within
`±0.1 Nm`, low torque variation, and fresh telemetry. The Collection tab shows
the specific gate that is still waiting.

## Training a force model

The Training tab is the preferred workflow:

1. Select a dataset directory and click **Audit dataset**.
2. Review planned/completed/skipped steps, accepted and rejected counts and
   reasons, samples per force/batch/repetition, loading/unloading coverage,
   quality distributions, step IDs, schemas, profiles, and independent split
   groups. Blocking audit errors disable training.
3. Choose a new output path, model name/version, validation fraction, ridge
   alpha, random seed, and leakage-safe split group.
4. Click **Train model**. Audit and fitting run in separate cancellable
   processes; **Cancel** terminates the child process without blocking the GUI.
5. Review training and validation MAE, RMSE, R², maximum absolute error,
   per-force validation metrics, sample/group counts, training and validation
   group identities, calibrated range, rejected rows, and compatibility fields.
6. Click **Accept and load model** only after review. Loading is validated in the
   camera worker against the current reference, geometry version, feature schema,
   and feature order. Failure keeps the prior model active. Successful models
   are recorded atomically in `models/model-registry.json`; **Load previous
   active model** provides rollback.

Existing model or metadata paths are refused unless **Explicitly approve
replacing existing artifacts** is checked. A trained candidate is never loaded
automatically, and activation does not automatically start inference.

After recording samples, train the included NumPy ridge model:

```bash
python train_force_model.py datasets/finger_calibration \
  --model-out models/finger-force.pkl
```

This creates:

```text
models/finger-force.pkl
models/finger-force.json
```

The JSON file stores model metadata, feature compatibility information, and
training/validation metrics. Existing model files are not overwritten unless
`--overwrite` is supplied.

For more training options:

```bash
python train_force_model.py --help
```

## Live force measurement

Start the camera application, load the same compatible reference profile, and
then enter:

```text
reference_load
tracking_start
model_load models/finger-force.pkl
model_info
force_start
force_status
```

Stop inference with:

```text
force_stop
```

The program rejects predictions when tracking, the origin, geometry, model
metadata, or the calibrated force range is invalid.

## Keyboard controls

These keys work while the camera window is focused:

| Key | Action |
| --- | --- |
| `q` or `Esc` | Quit |
| `a` | Open the blob-area tuning window |
| `m` | Toggle mask-only view |
| `c` | Toggle HSV calibration mode |
| `x` | Clear the active mouse-mode selection |
| `i` | Toggle two-click distance measurement |
| `s` | Save configuration and a diagnostic raw-frame image |

The `s` key does not record force-training samples. Use the `dataset_*`
commands for that.

## Calibration

### GUI click calibration

The Camera tab contains a **Click calibration** group that mirrors the OpenCV
`c`, `x`, and save workflow:

1. Start the camera and make sure the intended red marker is detected.
2. Click **Enable calibration**. This first applies the active HSV preset, just
   like pressing `c` in `main.py`.
3. Click the intended red blob in the live Camera image.
4. Inspect the sampled HSV, selected contour area, and calculated area range.
5. Use **Clear selected blob** to discard the selection, or click another blob
   while calibration remains enabled.
6. Click **Disable calibration** when finished.
7. Click **Save calibration** to atomically save the applied values to
   `camera-config.yaml`.

**Open tuning** switches to the Tuning tab, where the calculated thresholds can
be adjusted numerically. The equivalent console commands are
`calibration_start`, `calibration_stop`, `calibration_clear`, and
`calibration_status`.

The Calibration controls and the Tuning tab both provide a synchronized
**Mask-only preview** button. It uses the existing `show_mask_only on|off`
command and stays synchronized with the Runtime overlay checkbox.

The Calibration controls also contain live **Minimum area** and **Maximum area**
sliders with numeric spin boxes, both ranging from `0` to `1000 px²`. Dragging either slider immediately updates the
blob detector and synchronizes the values with Tuning. A maximum of `0` means
that no upper-area limit is applied. Slider changes are runtime values until
**Save calibration** is pressed.

### OpenCV keyboard calibration

If markers are missing or false blobs are detected:

1. Press `c` in the camera window.
2. Click a valid red blob.
3. Press `a` to adjust the accepted blob-area range if necessary.
4. Press `s` or enter `save_config`.
5. Press `c` again to leave calibration mode.

Use `show_mask_only on` to inspect the red mask and `show_circles on` to inspect
Hough-circle detections. Hough validation is optional in the current
configuration.

### Single blue-target calibration and tracking

The Camera workspace has a dedicated **Blue blob** tab. The blue detector and
its calibration values are independent of the red-marker pipeline.

1. Start the camera and open **Blue blob**.
2. Click **Enable blue calibration**.
3. Click the one intended blue blob in the live image. This samples its HSV
   value, updates the blue HSV and area limits, and binds the in-memory
   `BLUE_TARGET` track to that detection.
4. Click **Disable calibration**. Tracking continues after calibration mode is
   disabled.
5. Inspect **Image position** and **From ORIGIN (dx, dy)** in the same tab.
6. Use **Clear blue target** when a different physical target must be selected.
7. Click **Save blue calibration** to save the blue thresholds and tracking
   settings to `camera-config.yaml`.

Color detections correct the blue position on every frame. Lucas–Kanade optical
flow carries the point through short detection gaps, subject to the quality and
missing-frame limits under `blue_tracking`. Even if an extra blue candidate
appears, only the clicked target is tracked and redetections must remain within
its configured assignment distance.

The reported coordinate is:

```text
dx = blue_x - origin_x
dy = blue_y - origin_y
```

Coordinates are pixels in camera-image axes: positive X points right and
positive Y points down. A relative position is withheld whenever the blue track
or `REFERENCE_ORIGIN` is invalid. When the blue target is invalid, the tab and
status command explain whether color filtering, assignment distance, optical
flow, tracking quality, or the missing-frame limit caused the failure. The
equivalent console commands are
`blue_calibration_start`, `blue_calibration_stop`,
`blue_calibration_clear`, `blue_calibration_status`, and `blue_blob_status`.
Use `show_blue_mask_only on|off` for the blue mask preview.

## Configuration key behavior

- `geometry.structural_connections` is active and validated; it defines the
  length and angle feature contract. Changing it requires a compatible new
  reference/model workflow.
- `dataset.save_raw_coordinates`, `save_compensated_coordinates`, and
  `save_origin_relative_coordinates` are active. Disabled coordinate columns
  remain in schema version 3 but are blank. Selected-point and blue association
  metadata remains available for automated samples.
- `dataset.save_feature_names` must remain true because ordered names are part
  of the training compatibility contract.
- `dataset.frames_per_sample` controls the default manual batch count;
  `dataset.sample_interval_ms` controls spacing for manual batches. Protocol
  collection uses `collection_protocol.sample_interval_ms`.
- `force_model.model_path` and `metadata_path` are used by `model_load` when no
  arguments are supplied. Relative paths resolve from the Vision System
  directory.
- `force_model.minimum_calibrated_force_N`,
  `maximum_calibrated_force_N`, and `model_type` are retained for configuration
  compatibility but do not override a loaded artifact. Validated model metadata
  is authoritative.
- `tracking.minimum_valid_points` is retained for compatibility but is not used
  to weaken or replace the permanent-ID contract. Required point identities,
  reference origin, and geometry validity decide whether a frame is recordable.
- All motor keys in `motor-config.yaml` are validated and active. Unknown keys
  are rejected.

Compatibility-only settings are also exposed in the detached application
status instead of being silently ignored.

## Hardware-test checklist

The normal test suite must remain hardware-free. Before any separately approved
real-motor test:

1. Obtain explicit authorization for that hardware test.
2. Inspect the mechanism, cable routing, travel, pulley radius, efficiency, and
   special force-measurement finger.
3. Verify the physical emergency-stop and establish a safe exclusion zone.
4. Confirm CAN ID `21`, negative closing direction, the limits listed above,
   and `backend: hardware`.
5. Start with the gripper unloaded and open; keep the STOP MOTOR control visible.
6. Connect first and verify the drive remains disabled and feedback is plausible.
7. Enable and test stop/disable at zero torque before homing.
8. Home at only `-0.6 Nm`, then verify zero and the configured travel convention
   (hard bounds `-1..2.2 rad`, with the homed `0 rad` origin included).
9. Test position at reduced motion where appropriate, never exceeding `5 rad/s`
   and `20 rad/s²`.
10. Apply the smallest useful negative torque and verify the `0.5 Nm/s` ramp,
    measured torque, temperature, release, IDLE, and disable behavior.
11. Confirm the physical finger-force reading independently. Do not treat motor
    torque as newtons.
12. Return `motor-config.yaml` to `backend: mock` after the approved session.

## Common messages

### `Configured deforming total does not equal LINE_A + LINE_B`

The value of `tracking.expected_deforming_point_count` is different from:

```text
reference_ransac.expected_points_line_a
    + reference_ransac.expected_points_line_b
```

Correct the three values, restart the program, and create a new reference
profile.

### `Reference valid=False`

Read the `error:` lines printed below it. Common causes are:

- the configured line counts do not match the visible deforming markers;
- the origin was accidentally counted as a deforming marker;
- a crossing marker was counted in both lines;
- a red marker was not detected;
- a false red blob was detected as an extra marker;
- the two fitted lines are too close, too different in angle, or contain too
  many outliers.

The reported quality can still be high while the proposal is invalid because a
count or structural validation is a fatal requirement.

### `qt.qpa.events.reader: [heap]`

This is normally a harmless Qt/OpenCV window-system diagnostic, not a damaged
memory heap. If the window works normally, it can be ignored. Suppress it for
one run with:

```bash
QT_LOGGING_RULES="qt.qpa.events.reader.debug=false" python main.py
```

If it prints continuously with increasing numbers while the window freezes,
record several consecutive lines and investigate the GUI event loop.

## Automated tests

The full hardware-free suite, including offscreen Qt widget and worker tests, is
run with:

```bash
QT_QPA_PLATFORM=offscreen ../../virtual-env/bin/python -m pytest -q Test
```

### Lines are not visible

Run:

```text
show_lines on
reference_preview
```

The line overlay represents the current RANSAC proposal, so there is nothing to
draw before a preview. For loaded profiles, use the `LINE_A_*` and `LINE_B_*`
point labels and `show_geometry on`.

### Dataset samples are rejected

Run:

```text
tracking_status
dataset_status
```

Also inspect the warnings in the camera window and
`rejections.csv`. Recording requires a valid origin, valid tracking, and valid
geometry.

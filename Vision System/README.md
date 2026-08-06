# Vision System

This program detects red markers on a Fin Ray gripper, assigns permanent marker
IDs, tracks their deformation relative to a real moving origin marker, records
force-calibration datasets, trains a force model, and performs live force
prediction.

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

For 15 deforming markers split into groups of 6 and 9, use:

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
- `dataset_status` shows accepted, rejected, and pending counts.
- `dataset_stop` closes the active acquisition session.

Wait until the pending count reaches zero before changing the applied force.
Invalid frames are rejected and retried according to `camera-config.yaml`.

Data is written under:

```text
datasets/<experiment-id>/<acquisition-batch-id>/
├── metadata.yaml
├── samples.csv
├── rejections.csv          # present if attempts were rejected
└── images/                 # when save_images is enabled
```

The CSV contains the known force, raw and origin-relative coordinates, point
statuses, geometry quality, deterministic feature names, and feature values.

For a useful calibration dataset:

- include an unloaded `0 N` condition;
- record multiple known force levels over the intended operating range;
- collect many stable frames at every force;
- avoid moving or hiding the origin marker;
- use the same accepted reference profile throughout an experiment.

## Training a force model

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

If markers are missing or false blobs are detected:

1. Press `c` in the camera window.
2. Click a valid red blob.
3. Press `a` to adjust the accepted blob-area range if necessary.
4. Press `s` or enter `save_config`.
5. Press `c` again to leave calibration mode.

Use `show_mask_only on` to inspect the red mask and `show_circles on` to inspect
Hough-circle detections. Hough validation is optional in the current
configuration.

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


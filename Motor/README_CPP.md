# C++ Gripper Control for MAB Robotics Motor

This is a C++ port of the Python gripper control library for the MAB Robotics MA-P 45-10 motor using the CANdle SDK.

## Files

- **gripper.h** - Header file with Gripper class definition
- **gripper.cpp** - Implementation (stubs - requires CANdle SDK integration)
- **main.cpp** - Example usage demonstrating the API
- **CMakeLists.txt** - CMake build configuration (handles CANdle SDK linking)
- **build.sh** - Linux/macOS build script
- **build.bat** - Windows build script

## Building

### Quick Build (Recommended)

**Linux/macOS:**
```bash
cd /home/janek/Desktop/Programming/Programming/Motor
chmod +x build.sh
./build.sh
```

**Windows:**
```cmd
cd C:\Users\janek\Desktop\Programming\Programming\Motor
build.bat
```

### Manual Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Requirements

- CMake >= 3.16
- C++17 compatible compiler (gcc, clang, MSVC)
- CANdle SDK at `../../Repos/CANdle-SDK` (relative to this directory)

## Current Status

The C++ code is currently structured as **stubs/templates**. All CANdle SDK calls are commented out and marked with `TODO`.

To make this fully functional, you need to:

### 1. Integrate CANdle SDK Headers

Edit `gripper.cpp` and `gripper.h` to include actual CANdle SDK headers:

```cpp
#include <candle/candle.h>
#include <candle/MD.h>
```

Verify the correct header paths in your CANdle SDK installation.

### 2. Replace Placeholder API Calls

Uncomment and adjust all CANdle SDK calls in `gripper.cpp`. Example:

```cpp
// Current (stub):
// if (md.setMotionMode(MOTION_MODE_RAW_TORQUE) != OK) { ... }

// Should become (adjust based on actual API):
if (md.setMotionMode(MD_MotionMode_t::RAW_TORQUE) != MD_Error_t::OK) { ... }
```

### 3. Build

Run the build script or follow manual build steps above.

## Python Equivalent Functions

| C++ Function | Python Function | Purpose |
|---|---|---|
| `Gripper::connect()` | `connect()` | Initialize CANdle and MD |
| `Gripper::home()` | `home()` | Home motor to hard stop |
| `Gripper::wait_until_reached()` | `wait_until_reached()` | Wait for position |
| `Gripper::open_gripper()` | `open_gripper()` | Open gripper (position control) |
| `Gripper::close()` | `close()` | Close gripper (torque control) |
| `Gripper::close_debug()` | `close_debug()` | Debug with output |

## Key Differences from Python

1. **Error handling** - Uses exceptions instead of return codes
2. **Timing** - Uses `std::chrono` instead of Python `time` module
3. **Threading** - Uses `std::this_thread::sleep_for()` instead of `time.sleep()`
4. **Static methods** - Functions are static members of `Gripper` class
5. **Type system** - Explicit types (double, int) instead of Python's dynamic typing

## Example Usage

```cpp
CANdle candle;
MD md;
Gripper::connect(candle, md);
Gripper::home(md);
Gripper::open_gripper(md);
Gripper::close(md, -1.0);
```

## Notes

- CAN_ID is hardcoded to 21 (matches Python version)
- All timeout values default to 5 seconds
- Torque values follow the same sign convention as Python (negative = close)
- The implementation assumes CANdle SDK follows similar API patterns to the Python bindings

## CMake Configuration

The `CMakeLists.txt` automatically:
- Detects the CANdle SDK at `../../Repos/CANdle-SDK`
- Builds required CANdle libraries as subdirectories
- Creates a static library `gripper_lib` for the gripper control code
- Links the executable `gripper_control` to all dependencies
- Generates compile commands for IDE support

**Key CMake variables:**
- `CANDLE_SDK_PATH` - Path to CANdle SDK (default: `../../Repos/CANdle-SDK`)
- `CMAKE_BUILD_TYPE` - Build type (Debug/Release)
- `CMAKE_EXPORT_COMPILE_COMMANDS` - Generate compile_commands.json for IDEs

## Troubleshooting

### CMake can't find CANdle SDK
- Verify `../../Repos/CANdle-SDK` exists and is accessible
- Or set `CANDLE_SDK_PATH` explicitly:
  ```bash
  cmake .. -DCANDLE_SDK_PATH=/path/to/CANdle-SDK
  ```

### Linker errors
- Ensure all CANdle SDK subdirectories are built: check `CMakeLists.txt` subdirectory declarations
- Verify compiler compatibility with CANdle SDK requirements

### Build fails with "CAN frame too long" warning
- This is a runtime issue, not a build issue
- See Python version for context on impedance control mode issues

## Reference

- CANdle SDK: `../../Repos/CANdle-SDK/`
- Original Python: `gripper.py`
- MD Python Bindings: `md_python_bindings.md`


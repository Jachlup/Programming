"""
Parallel-jaw, belt-driven gripper control functions.
Built on top of pyCandle (CANdle-SDK) MD bindings for the MAB Robotics MA-P 45-10.
"""

import pyCandle

CAN_ID = 21


def connect():
    """Attach the CANdle dongle and initialize the MD at CAN_ID. Returns (candle, md)."""
    candle = pyCandle.attachCandle(pyCandle.CANdleDatarate_E.CAN_DATARATE_1M, pyCandle.busTypes_t.USB)
    md = pyCandle.MD(CAN_ID, candle)

    error = md.init()
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"MD init failed: {error}")

    error = md.clearErrors()
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"clearErrors failed: {error}")

    error = md.enable()
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"enable failed: {error}")

    return candle, md

def home(md, torque=-0.6, epsilon=0.01, stall_time_s=0.1, timeout_s=5.0):
    """Push the jaw toward the hard stop under constant torque, detect the stall, then zero the encoder."""
    import time

    error = md.setMotionMode(pyCandle.MotionMode_t.RAW_TORQUE)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"set RAW_TORQUE failed: {error}")

    error = md.setTargetTorque(torque)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"setTargetTorque failed: {error}")

    start = time.time()
    stall_start = None
    last_position, _ = md.getPosition()
    hit_stop = False

    while time.time() - start < timeout_s:
        time.sleep(0.02)
        position, _ = md.getPosition()

        if abs(position - last_position) < epsilon:
            if stall_start is None:
                stall_start = time.time()
            elif time.time() - stall_start >= stall_time_s:
                hit_stop = True
                break
        else:
            stall_start = None

        last_position = position

    md.setTargetTorque(0.0)

    if not hit_stop:
        raise RuntimeError("Homing timed out before detecting the stall")

    error = md.zero()
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"zero() failed: {error}")


def wait_until_reached(md, target, tolerance=0.5, timeout_s=5.0,
                        feedback_cb=None, cancel_check=None):
    """
    Block until the motor's position is within tolerance of target, or timeout.

    feedback_cb: optional callable(position) — called every loop iteration.
                 In ROS2, pass goal_handle.publish_feedback wrapped to build the msg.
    cancel_check: optional callable() -> bool — if it returns True, abort early.
                  In ROS2, pass goal_handle.is_cancel_requested.
    """
    import time

    start = time.time()
    while time.time() - start < timeout_s:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Motion cancelled")

        position, error = md.getPosition()
        if error != pyCandle.MD_Error_t.OK:
            raise RuntimeError(f"getPosition failed: {error}")

        if feedback_cb is not None:
            feedback_cb(position)

        if abs(position - target) <= tolerance:
            return

        time.sleep(0.02)

    raise RuntimeError("Timed out waiting to reach target position")

def open_gripper(md, position=2.0, velocity=20.0, accel=40.0,
                  feedback_cb=None, cancel_check=None):
    """Move the jaw to the open position using profiled position control, and block until it arrives."""
    error = md.setMotionMode(pyCandle.MotionMode_t.PROFILE_POSITION)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"set PROFILE_POSITION failed: {error}")

    error = md.setProfileVelocity(velocity)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"setProfileVelocity failed: {error}")

    error = md.setProfileAcceleration(accel)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"setProfileAcceleration failed: {error}")

    error = md.setTargetPosition(position)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"setTargetPosition failed: {error}")

    wait_until_reached(md, position, feedback_cb=feedback_cb, cancel_check=cancel_check)

def close(md, torque=-1.0):
    """Close with constant motor torque in Nm. The caller must later call stop()."""
    error = md.setMotionMode(pyCandle.MotionMode_t.RAW_TORQUE)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"set RAW_TORQUE failed: {error}")

    error = md.setTargetTorque(torque)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"setTargetTorque failed: {error}")


def stop(md):
    """Release commanded torque and put the controller in idle mode."""
    error = md.setTargetTorque(0.0)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"setTargetTorque(0) failed: {error}")

    error = md.setMotionMode(pyCandle.MotionMode_t.IDLE)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"set IDLE failed: {error}")

def close_debug(md, torque=1.0, epsilon=0.01, stall_time_s=0.1, timeout_s=5.0):
    import time

    md.setMotionMode(pyCandle.MotionMode_t.RAW_TORQUE)
    md.setTargetTorque(torque)

    start = time.time()
    last_position, _ = md.getPosition()

    while time.time() - start < timeout_s:
        time.sleep(0.02)
        position, _ = md.getPosition()
        current_torque, _ = md.getTorque()
        print(f"t={time.time()-start:.2f}  pos={position:.4f}  torque={current_torque:.3f}  delta={abs(position-last_position):.5f}")
        last_position = position

    md.setTargetTorque(0.0)

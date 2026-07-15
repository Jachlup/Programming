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

def home(md, velocity=-5.0, torque_threshold=0.4, timeout_s=5.0):
    """Drive the jaw toward the hard stop, stop on a torque spike, then zero the encoder."""
    import time

    error = md.setMotionMode(pyCandle.MotionMode_t.VELOCITY_PID)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"set VELOCITY_PID failed: {error}")

    error = md.setTargetVelocity(velocity)
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"setTargetVelocity failed: {error}")

    start = time.time()
    hit_stop = False
    while time.time() - start < timeout_s:
        if abs(md.getTorque()) >= torque_threshold:
            hit_stop = True
            break
        time.sleep(0.01)

    md.setTargetVelocity(0.0)

    if not hit_stop:
        raise RuntimeError("Homing timed out before hitting the hard stop")

    error = md.zero()
    if error != pyCandle.MD_Error_t.OK:
        raise RuntimeError(f"zero() failed: {error}")
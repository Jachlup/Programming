import gripper
import time

if __name__ == "__main__":
    candle, md = gripper.connect()
    gripper.home(md)
    print("Homed. Position:", md.getPosition())
    gripper.open_gripper(md) 
    """
    gripper.open_gripper(
    md,
    feedback_cb=lambda pos: goal_handle.publish_feedback(Feedback(current_position=pos)),
    cancel_check=goal_handle.is_cancel_requested,
    )
    """
    time.sleep(1.0)
    # setTargetTorque expects Nm, not amperes. close() also selects RAW_TORQUE.
    gripper.close(md, torque=-0.2)
    print("Holding with -0.2 Nm. Press Ctrl+C to release.")
    try:
        next_report = time.monotonic()
        while True:
            gripper.maintain_close(md, -0.2)
            now = time.monotonic()
            if now >= next_report:
                actual_torque, torque_error = md.getTorque()
                position, position_error = md.getPosition()
                if torque_error != gripper.pyCandle.MD_Error_t.OK:
                    raise RuntimeError(f"getTorque failed: {torque_error}")
                if position_error != gripper.pyCandle.MD_Error_t.OK:
                    raise RuntimeError(f"getPosition failed: {position_error}")
                print(f"target=-0.200 Nm, actual={actual_torque:.3f} Nm, position={position:.3f} rad")
                next_report = now + 0.5
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nReleasing gripper.")
    finally:
        gripper.stop(md)
        md.disable()

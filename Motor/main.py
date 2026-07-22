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
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nReleasing gripper.")
    finally:
        gripper.stop(md)
        md.disable()

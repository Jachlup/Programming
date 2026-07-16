import gripper
import pyCandle as pc
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
    md.setTargetTorque(0.0) 
    pc.MotionMode_t.RAW_TORQUE
    md.setMaxTorque(2.0)
    md.setTargetTorque(0.2)


    time.sleep(1.0)

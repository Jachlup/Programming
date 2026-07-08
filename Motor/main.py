import time
import pyCandle as pc


candle = pc.attachCandle(
    pc.CANdleDatarate_E.CAN_DATARATE_1M,
    pc.busTypes_t.USB
)

md = pc.MD(21, candle)

err = md.init()

if err == pc.MD_Error_t.OK:

    md.setMotionMode(
        pc.MotionMode_t.VELOCITY_PID
    )

    md.enable()

    try:
        target_velocity = 1.0   # rad/s

        while True:

            # command
            md.setTargetVelocity(target_velocity)

            # feedback
            velocity = md.getVelocity()
            position = md.getPosition()

            print(
                f"Velocity: {velocity[0]:.3f} rad/s"
            )

            print(
                f"Position: {position[0]:.3f} rad"
            )

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass

    finally:
        md.disable()
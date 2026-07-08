import pyCandle as pc


class MABMotor:

    def __init__(self, motor_id):

        print("Connecting CANdle...")

        self.candle = pc.attachCandle(
            pc.CANdleDatarate_E.CAN_DATARATE_1M,
            pc.busTypes_t.USB
        )

        print("Creating motor object...")

        self.motor = pc.MD(
            motor_id,
            self.candle
        )

        print("Initializing motor...")

        error = self.motor.init()

        if error != pc.MD_Error_t.OK:
            raise Exception(
                f"Motor initialization failed: {error}"
            )

        print("Motor ready")


    def enable(self):
        self.motor.enable()


    def disable(self):
        self.motor.disable()
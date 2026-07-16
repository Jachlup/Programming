#ifndef GRIPPER_H
#define GRIPPER_H

#include <chrono>
#include <stdexcept>

// Forward declarations (adjust includes based on CANdle SDK structure)
class MD;
class CANdle;

class Gripper {
public:
    static constexpr int CAN_ID = 21;
    
    /**
     * Attach the CANdle dongle and initialize the MD at CAN_ID.
     */
    static void connect(CANdle& candle, MD& md);
    
    /**
     * Push the jaw toward the hard stop under constant torque, detect the stall, 
     * then zero the encoder.
     */
    static void home(
        MD& md,
        double torque = -0.6,
        double epsilon = 0.01,
        double stall_time_s = 0.1,
        double timeout_s = 5.0
    );
    
    /**
     * Block until the motor's position is within tolerance of target, or timeout.
     */
    static void wait_until_reached(
        MD& md,
        double target,
        double tolerance = 0.5,
        double timeout_s = 5.0
    );
    
    /**
     * Move the jaw to the open position using profiled position control.
     */
    static void open_gripper(
        MD& md,
        double position = 2.0,
        double velocity = 20.0,
        double accel = 40.0
    );
    
    /**
     * Apply constant torque to close the gripper.
     */
    static void close(
        MD& md,
        double torque = -1.0
    );
    
    /**
     * Debug version that prints position and torque every iteration.
     */
    static void close_debug(
        MD& md,
        double torque = 1.0,
        double epsilon = 0.01,
        double stall_time_s = 0.1,
        double timeout_s = 5.0
    );
};

#endif // GRIPPER_H

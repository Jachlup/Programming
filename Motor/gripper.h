#ifndef GRIPPER_H
#define GRIPPER_H

#include <chrono>
#include <stdexcept>

// Forward declarations for CANdle SDK
namespace mab {
    class Candle;
    class MD;
}

class Gripper {
public:
    static constexpr int CAN_ID = 21;
    
    /**
     * Attach the CANdle dongle and initialize one MD. The drive stays disabled
     * unless enable_drive is explicitly requested.
     */
    static void connect(
        mab::Candle*& candle,
        mab::MD*& md,
        int can_id = CAN_ID,
        bool clear_faults = false,
        bool enable_drive = false
    );

    /** Release the MD and CANdle objects created by connect(). */
    static void disconnect(mab::Candle*& candle, mab::MD*& md) noexcept;

    static void enable(mab::MD* md);
    static void disable(mab::MD* md);
    static void clear_faults(mab::MD* md);
    static void zero_position(mab::MD* md);
    static void set_current_limit(mab::MD* md, double current_A);
    static void set_maximum_torque(mab::MD* md, double torque_Nm);

    /** Start a profiled position move without blocking for completion. */
    static void start_profiled_position(
        mab::MD* md,
        double position,
        double velocity,
        double acceleration
    );

    static double read_position(mab::MD* md);
    static double read_velocity(mab::MD* md);
    static double read_torque(mab::MD* md);
    static double read_temperature(mab::MD* md);
    
    /**
     * Push the jaw toward the hard stop under constant torque, detect the stall, 
     * then zero the encoder.
     */
    static void home(
        mab::MD* md,
        double torque = -0.6,
        double epsilon = 0.01,
        double stall_time_s = 0.1,
        double timeout_s = 5.0
    );
    
    /**
     * Block until the motor's position is within tolerance of target, or timeout.
     */
    static void wait_until_reached(
        mab::MD* md,
        double target,
        double tolerance = 0.5,
        double timeout_s = 5.0
    );
    
    /**
     * Move the jaw to the open position using profiled position control.
     */
    static void open_gripper(
        mab::MD* md,
        double position = 2.0,
        double velocity = 5.0,
        double accel = 20.0
    );
    
    /**
     * Apply constant torque to close the gripper.
     */
    static void close(
        mab::MD* md,
        double torque = -1.0
    );

    /** Re-send the torque target. Call periodically while holding. */
    static void maintain_close(mab::MD* md, double torque);

    /** Stop producing torque and put the controller in idle mode. */
    static void stop(mab::MD* md);
    
    /**
     * Debug version that prints position and torque every iteration.
     */
    static void close_debug(
        mab::MD* md,
        double torque = 1.0,
        double epsilon = 0.01,
        double stall_time_s = 0.1,
        double timeout_s = 5.0
    );
};

#endif // GRIPPER_H

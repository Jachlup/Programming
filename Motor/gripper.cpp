#include "gripper.h"
#include <thread>
#include <iostream>
#include <cmath>

// Include CANdle SDK headers (adjust paths based on your installation)
// #include <candle/candle.h>
// #include <candle/MD.h>

void Gripper::connect(CANdle& candle, MD& md) {
    // Initialize CANdle dongle at 1M baud via USB
    // candle.attachCandle(CANDLE_DATARATE_1M, CANDLE_BUS_USB);
    
    // Initialize MD at CAN_ID
    // md.init(CAN_ID, candle);
    
    // TODO: Replace with actual CANdle SDK calls
    throw std::runtime_error("connect() requires CANdle SDK integration");
}

void Gripper::home(
    MD& md,
    double torque,
    double epsilon,
    double stall_time_s,
    double timeout_s
) {
    // Set raw torque mode
    // if (md.setMotionMode(MOTION_MODE_RAW_TORQUE) != OK) {
    //     throw std::runtime_error("set RAW_TORQUE failed");
    // }
    
    // Apply target torque
    // if (md.setTargetTorque(torque) != OK) {
    //     throw std::runtime_error("setTargetTorque failed");
    // }
    
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> stall_start;
    bool stall_detected = false;
    double last_position = 0.0;
    // last_position = md.getPosition().first;
    
    while (true) {
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        if (std::chrono::duration<double>(elapsed).count() > timeout_s) {
            // md.setTargetTorque(0.0);
            throw std::runtime_error("Homing timed out before detecting the stall");
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        
        // double position = md.getPosition().first;
        double position = 0.0; // TODO: get actual position
        
        if (std::abs(position - last_position) < epsilon) {
            if (!stall_detected) {
                stall_start = std::chrono::high_resolution_clock::now();
                stall_detected = true;
            } else {
                auto stall_elapsed = std::chrono::high_resolution_clock::now() - stall_start;
                if (std::chrono::duration<double>(stall_elapsed).count() >= stall_time_s) {
                    break;  // Stall detected
                }
            }
        } else {
            stall_detected = false;
        }
        
        last_position = position;
    }
    
    // md.setTargetTorque(0.0);
    // if (md.zero() != OK) {
    //     throw std::runtime_error("zero() failed");
    // }
}

void Gripper::wait_until_reached(
    MD& md,
    double target,
    double tolerance,
    double timeout_s
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    while (true) {
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        if (std::chrono::duration<double>(elapsed).count() > timeout_s) {
            throw std::runtime_error("Timed out waiting to reach target position");
        }
        
        // double position = md.getPosition().first;
        double position = 0.0;  // TODO: get actual position
        
        if (std::abs(position - target) <= tolerance) {
            return;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void Gripper::open_gripper(
    MD& md,
    double position,
    double velocity,
    double accel
) {
    // Set position PID mode
    // if (md.setMotionMode(MOTION_MODE_POSITION_PID) != OK) {
    //     throw std::runtime_error("set POSITION_PID failed");
    // }
    
    // Set profile parameters
    // if (md.setProfileVelocity(velocity) != OK) {
    //     throw std::runtime_error("setProfileVelocity failed");
    // }
    
    // if (md.setProfileAcceleration(accel) != OK) {
    //     throw std::runtime_error("setProfileAcceleration failed");
    // }
    
    // Set target position
    // if (md.setTargetPosition(position) != OK) {
    //     throw std::runtime_error("setTargetPosition failed");
    // }
    
    // Wait until position is reached
    wait_until_reached(md, position, 0.5, 5.0);
}

void Gripper::close(MD& md, double torque) {
    // Set raw torque mode
    // if (md.setMotionMode(MOTION_MODE_RAW_TORQUE) != OK) {
    //     throw std::runtime_error("set RAW_TORQUE failed");
    // }
    
    // Apply target torque
    // if (md.setTargetTorque(torque) != OK) {
    //     throw std::runtime_error("setTargetTorque failed");
    // }
}

void Gripper::close_debug(
    MD& md,
    double torque,
    double epsilon,
    double stall_time_s,
    double timeout_s
) {
    // Set raw torque mode
    // md.setMotionMode(MOTION_MODE_RAW_TORQUE);
    // md.setTargetTorque(torque);
    
    auto start = std::chrono::high_resolution_clock::now();
    double last_position = 0.0;
    // last_position = md.getPosition().first;
    
    while (true) {
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        if (std::chrono::duration<double>(elapsed).count() > timeout_s) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        
        // double position = md.getPosition().first;
        // double current_torque = md.getTorque().first;
        double position = 0.0;
        double current_torque = 0.0;
        
        double delta = std::abs(position - last_position);
        std::printf("t=%.2f  pos=%.4f  torque=%.3f  delta=%.5f\n",
                    std::chrono::duration<double>(elapsed).count(),
                    position,
                    current_torque,
                    delta);
        
        last_position = position;
    }
    
    // md.setTargetTorque(0.0);
}

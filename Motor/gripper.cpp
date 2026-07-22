#include "gripper.h"
#include <thread>
#include <iostream>
#include <cmath>
#include <string>

// Include CANdle SDK headers
#include "candle.hpp"
#include "MD.hpp"

namespace {
void require_ok(mab::MD::Error_t error, const char* operation) {
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}

float read_position(mab::MD* md) {
    const auto [position, error] = md->getPosition();
    require_ok(error, "getPosition");
    return position;
}
} // namespace

void Gripper::connect(mab::Candle*& candle, mab::MD*& md) {
    // Initialize CANdle dongle at 1M baud via USB
    candle = mab::attachCandle(
        mab::CANdleDatarate_E::CAN_DATARATE_1M,
        mab::candleTypes::busTypes_t::USB
    );
    
    if (!candle) {
        throw std::runtime_error("Failed to attach CANdle");
    }
    
    // Connect directly to the known motor. Scanning the CAN bus is unnecessary.
    md = new mab::MD(CAN_ID, candle);
    
    require_ok(md->init(), "MD init");
    require_ok(md->clearErrors(), "clearErrors");
    // Motion mode is volatile and must be selected again after power-up/reset.
    require_ok(md->enable(), "enable");
}

void Gripper::home(
    mab::MD* md,
    double torque,
    double epsilon,
    double stall_time_s,
    double timeout_s
) {
    // Set raw torque mode
    auto error = md->setMotionMode(mab::MdMode_E::RAW_TORQUE);
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("set RAW_TORQUE failed");
    }
    
    // Apply target torque
    error = md->setTargetTorque(static_cast<float>(torque));
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("setTargetTorque failed");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> stall_start;
    bool stall_detected = false;
    
    double last_position = read_position(md);
    
    while (true) {
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        if (std::chrono::duration<double>(elapsed).count() > timeout_s) {
            md->setTargetTorque(0.0f);
            throw std::runtime_error("Homing timed out before detecting the stall");
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        
        double position = read_position(md);
        
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
    
    require_ok(md->setTargetTorque(0.0f), "release homing torque");
    error = md->zero();
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("zero() failed");
    }
}

void Gripper::wait_until_reached(
    mab::MD* md,
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
        
        double position = read_position(md);
        
        if (std::abs(position - target) <= tolerance) {
            return;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void Gripper::open_gripper(
    mab::MD* md,
    double position,
    double velocity,
    double accel
) {
    // Profile mode is required for profile velocity/acceleration to take effect.
    auto error = md->setMotionMode(mab::MdMode_E::POSITION_PROFILE);
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("set POSITION_PROFILE failed");
    }
    
    // Set profile parameters
    error = md->setProfileVelocity(static_cast<float>(velocity));
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("setProfileVelocity failed");
    }
    
    error = md->setProfileAcceleration(static_cast<float>(accel));
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("setProfileAcceleration failed");
    }
    
    // Set target position
    error = md->setTargetPosition(static_cast<float>(position));
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("setTargetPosition failed");
    }
    
    // Wait until position is reached
    wait_until_reached(md, position, 0.5, 5.0);
}

void Gripper::close(mab::MD* md, double torque) {
    // Set raw torque mode
    auto error = md->setMotionMode(mab::MdMode_E::RAW_TORQUE);
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("set RAW_TORQUE failed");
    }
    
    // Apply target torque
    error = md->setTargetTorque(static_cast<float>(torque));
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("setTargetTorque failed");
    }
}

void Gripper::maintain_close(mab::MD* md, double torque) {
    // Some drive/firmware configurations use a command watchdog. Refreshing the
    // fixed target also makes holding robust against a lost CAN frame.
    require_ok(md->setTargetTorque(static_cast<float>(torque)), "refresh target torque");
}

void Gripper::stop(mab::MD* md) {
    require_ok(md->setTargetTorque(0.0f), "set zero torque");
    require_ok(md->setMotionMode(mab::MdMode_E::IDLE), "set IDLE");
}

void Gripper::close_debug(
    mab::MD* md,
    double torque,
    double epsilon,
    double stall_time_s,
    double timeout_s
) {
    (void)epsilon;
    (void)stall_time_s;
    // Set raw torque mode
    auto error = md->setMotionMode(mab::MdMode_E::RAW_TORQUE);
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("set RAW_TORQUE failed");
    }
    
    error = md->setTargetTorque(static_cast<float>(torque));
    if (error != mab::MD::Error_t::OK) {
        throw std::runtime_error("setTargetTorque failed");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    double last_position = read_position(md);
    
    while (true) {
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        if (std::chrono::duration<double>(elapsed).count() > timeout_s) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        
        auto position_result = md->getPosition();
        auto torque_result = md->getTorque();
        require_ok(position_result.second, "getPosition");
        require_ok(torque_result.second, "getTorque");
        double position = position_result.first;
        double current_torque = torque_result.first;
        
        double delta = std::abs(position - last_position);
        std::printf("t=%.2f  pos=%.4f  torque=%.3f  delta=%.5f\n",
                    std::chrono::duration<double>(elapsed).count(),
                    position,
                    current_torque,
                    delta);
        
        last_position = position;
    }
    
    md->setTargetTorque(0.0f);
}

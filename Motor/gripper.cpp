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

template <typename Value>
double checked_read(
    const std::pair<Value, mab::MD::Error_t>& result,
    const char* operation
) {
    require_ok(result.second, operation);
    const double value = static_cast<double>(result.first);
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(operation) + " returned a non-finite value");
    }
    return value;
}
} // namespace

void Gripper::connect(
    mab::Candle*& candle,
    mab::MD*& md,
    int can_id,
    bool clear_faults,
    bool enable_drive
) {
    if (candle != nullptr || md != nullptr) {
        throw std::runtime_error("Gripper is already connected");
    }
    if (can_id < 1 || can_id > 127) {
        throw std::invalid_argument("CAN ID must be in 1..127");
    }

    // Initialize CANdle dongle at 1M baud via USB
    candle = mab::attachCandle(
        mab::CANdleDatarate_E::CAN_DATARATE_1M,
        mab::candleTypes::busTypes_t::USB
    );
    
    if (!candle) {
        throw std::runtime_error("Failed to attach CANdle");
    }
    
    try {
        // Connect directly to the configured motor. Scanning is unnecessary.
        md = new mab::MD(static_cast<mab::canId_t>(can_id), candle);
        require_ok(md->init(), "MD init");
        if (clear_faults) {
            require_ok(md->clearErrors(), "clearErrors");
        }
        if (enable_drive) {
            require_ok(md->enable(), "enable");
        }
    } catch (...) {
        if (md != nullptr) {
            md->setTargetTorque(0.0f);
            md->setMotionMode(mab::MdMode_E::IDLE);
            md->disable();
        }
        disconnect(candle, md);
        throw;
    }
}

void Gripper::disconnect(mab::Candle*& candle, mab::MD*& md) noexcept {
    delete md;
    md = nullptr;
    mab::detachCandle(candle);
    candle = nullptr;
}

void Gripper::enable(mab::MD* md) {
    require_ok(md->enable(), "enable");
}

void Gripper::disable(mab::MD* md) {
    require_ok(md->disable(), "disable");
}

void Gripper::clear_faults(mab::MD* md) {
    require_ok(md->clearErrors(), "clearErrors");
}

void Gripper::zero_position(mab::MD* md) {
    require_ok(md->zero(), "zero");
}

void Gripper::set_current_limit(mab::MD* md, double current_A) {
    if (!std::isfinite(current_A) || current_A <= 0.0) {
        throw std::invalid_argument("Current limit must be finite and positive");
    }
    require_ok(md->setCurrentLimit(static_cast<float>(current_A)), "setCurrentLimit");
}

void Gripper::set_maximum_torque(mab::MD* md, double torque_Nm) {
    if (!std::isfinite(torque_Nm) || torque_Nm <= 0.0) {
        throw std::invalid_argument("Maximum torque must be finite and positive");
    }
    require_ok(md->setMaxTorque(static_cast<float>(torque_Nm)), "setMaxTorque");
}

void Gripper::start_profiled_position(
    mab::MD* md,
    double position,
    double velocity,
    double acceleration
) {
    if (!std::isfinite(position) || !std::isfinite(velocity) ||
        !std::isfinite(acceleration) || velocity <= 0.0 || acceleration <= 0.0) {
        throw std::invalid_argument("Profile values must be finite and rates must be positive");
    }
    require_ok(md->setMotionMode(mab::MdMode_E::POSITION_PROFILE), "set POSITION_PROFILE");
    require_ok(md->setProfileVelocity(static_cast<float>(velocity)), "setProfileVelocity");
    require_ok(
        md->setProfileAcceleration(static_cast<float>(acceleration)),
        "setProfileAcceleration"
    );
    require_ok(md->setTargetPosition(static_cast<float>(position)), "setTargetPosition");
}

double Gripper::read_position(mab::MD* md) {
    return checked_read(md->getPosition(), "getPosition");
}

double Gripper::read_velocity(mab::MD* md) {
    return checked_read(md->getVelocity(), "getVelocity");
}

double Gripper::read_torque(mab::MD* md) {
    return checked_read(md->getTorque(), "getTorque");
}

double Gripper::read_temperature(mab::MD* md) {
    return checked_read(md->getTemperature(), "getTemperature");
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
    start_profiled_position(md, position, velocity, accel);
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

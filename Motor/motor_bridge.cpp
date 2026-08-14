#include "gripper.h"

#include "MD.hpp"

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {
constexpr const char* PROTOCOL_PREFIX = "MOTOR_BRIDGE";
constexpr int PROTOCOL_VERSION = 1;
constexpr double HARD_MAX_CURRENT_A = 0.5;
constexpr double HARD_MAX_CONTROLLER_TORQUE_NM = 10.0;
constexpr double HARD_MIN_POSITION_RAD = -1.0;
constexpr double HARD_MAX_POSITION_RAD = 2.2;
constexpr double HARD_MAX_VELOCITY_RAD_S = 5.0;
constexpr double HARD_MAX_ACCELERATION_RAD_S2 = 20.0;
constexpr int HARD_CLOSING_DIRECTION = -1;
constexpr double HARD_MAX_CLOSING_TORQUE_NM = 4.0;
constexpr double HARD_MAX_TEMPERATURE_C = 50.0;

std::atomic_bool shutdown_requested{false};

extern "C" void request_shutdown(int) {
    shutdown_requested.store(true);
#ifndef _WIN32
    // close() is async-signal-safe and wakes a process blocked in getline().
    ::close(STDIN_FILENO);
#endif
}

struct SafetyLimits {
    double current_A = 0.0;
    double controller_torque_Nm = 0.0;
    double minimum_position_rad = 0.0;
    double maximum_position_rad = 0.0;
    double maximum_velocity_rad_s = 0.0;
    double maximum_acceleration_rad_s2 = 0.0;
    int closing_direction = 0;
    double maximum_closing_torque_Nm = 0.0;
    double maximum_temperature_C = 0.0;
    bool configured = false;
    bool current_applied = false;
    bool torque_limit_applied = false;
};

double read_number(std::istringstream& input, const char* name) {
    double value = 0.0;
    if (!(input >> value) || !std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be a finite number");
    }
    return value;
}

int read_integer(std::istringstream& input, const char* name) {
    int value = 0;
    if (!(input >> value)) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return value;
}

void require_end(std::istringstream& input) {
    std::string extra;
    if (input >> extra) {
        throw std::invalid_argument("Unexpected command argument: " + extra);
    }
}

void require_configured(const SafetyLimits& limits) {
    if (!limits.configured) {
        throw std::runtime_error("Safety limits have not been configured");
    }
}

void require_motion_ready(const SafetyLimits& limits, bool enabled) {
    require_configured(limits);
    if (!limits.current_applied || !limits.torque_limit_applied) {
        throw std::runtime_error("Controller current and torque limits must be applied first");
    }
    if (!enabled) {
        throw std::runtime_error("Drive is disabled");
    }
}

void validate_torque(const SafetyLimits& limits, double torque_Nm) {
    require_configured(limits);
    if (torque_Nm == 0.0) {
        return;
    }
    const int sign = std::signbit(torque_Nm) ? -1 : 1;
    if (sign != limits.closing_direction) {
        throw std::invalid_argument("Torque does not use the configured closing direction");
    }
    if (std::abs(torque_Nm) > limits.maximum_closing_torque_Nm) {
        throw std::invalid_argument("Torque exceeds the configured closing limit");
    }
}

std::string clean_error(std::string message) {
    for (char& character : message) {
        if (character == '\n' || character == '\r') {
            character = ' ';
        }
    }
    return message;
}

void reply_ok(const std::string& payload = {}) {
    std::cout << PROTOCOL_PREFIX << " OK";
    if (!payload.empty()) {
        std::cout << ' ' << payload;
    }
    std::cout << '\n' << std::flush;
}

void reply_error(const std::string& message) {
    std::cout << PROTOCOL_PREFIX << " ERR " << clean_error(message) << '\n' << std::flush;
}

void shutdown_drive(mab::MD* md, bool& enabled) {
    if (md == nullptr) {
        enabled = false;
        return;
    }
    std::exception_ptr first_failure;
    try {
        Gripper::stop(md);
    } catch (...) {
        first_failure = std::current_exception();
    }
    try {
        Gripper::disable(md);
    } catch (...) {
        if (first_failure == nullptr) {
            first_failure = std::current_exception();
        }
    }
    enabled = false;
    if (first_failure != nullptr) {
        std::rethrow_exception(first_failure);
    }
}

void safe_shutdown(mab::MD* md, bool& enabled) noexcept {
    try {
        shutdown_drive(md, enabled);
    } catch (...) {
    }
}

void configure_limits(std::istringstream& input, SafetyLimits& limits, bool enabled) {
    if (enabled) {
        throw std::runtime_error("Disable the drive before configuring safety limits");
    }
    SafetyLimits requested;
    requested.current_A = read_number(input, "current limit");
    requested.controller_torque_Nm = read_number(input, "controller torque limit");
    requested.minimum_position_rad = read_number(input, "minimum position");
    requested.maximum_position_rad = read_number(input, "maximum position");
    requested.maximum_velocity_rad_s = read_number(input, "maximum velocity");
    requested.maximum_acceleration_rad_s2 = read_number(input, "maximum acceleration");
    requested.closing_direction = read_integer(input, "closing direction");
    requested.maximum_closing_torque_Nm = read_number(input, "closing torque limit");
    requested.maximum_temperature_C = read_number(input, "maximum temperature");
    require_end(input);

    if (requested.current_A <= 0.0 || requested.controller_torque_Nm <= 0.0 ||
        requested.minimum_position_rad >= requested.maximum_position_rad ||
        requested.minimum_position_rad > 0.0 ||
        requested.maximum_position_rad < 0.0 ||
        requested.maximum_velocity_rad_s <= 0.0 ||
        requested.maximum_acceleration_rad_s2 <= 0.0 ||
        (requested.closing_direction != -1 && requested.closing_direction != 1) ||
        requested.maximum_closing_torque_Nm <= 0.0 ||
        requested.maximum_closing_torque_Nm > requested.controller_torque_Nm ||
        requested.maximum_temperature_C <= 0.0) {
        throw std::invalid_argument("Invalid safety-limit configuration");
    }
    if (requested.current_A > HARD_MAX_CURRENT_A ||
        requested.controller_torque_Nm > HARD_MAX_CONTROLLER_TORQUE_NM ||
        requested.minimum_position_rad < HARD_MIN_POSITION_RAD ||
        requested.maximum_position_rad > HARD_MAX_POSITION_RAD ||
        requested.maximum_velocity_rad_s > HARD_MAX_VELOCITY_RAD_S ||
        requested.maximum_acceleration_rad_s2 > HARD_MAX_ACCELERATION_RAD_S2 ||
        requested.closing_direction != HARD_CLOSING_DIRECTION ||
        requested.maximum_closing_torque_Nm > HARD_MAX_CLOSING_TORQUE_NM ||
        requested.maximum_temperature_C > HARD_MAX_TEMPERATURE_C) {
        throw std::invalid_argument("Safety configuration exceeds the fixed gripper limits");
    }
    requested.configured = true;
    limits = requested;
}

int parse_can_id(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 1 || value > 127) {
        throw std::invalid_argument("CAN ID must be an integer in 1..127");
    }
    return static_cast<int>(value);
}

void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " <can_id>\n"
        << "Persistent line protocol for the Vision System C++ motor backend.\n"
        << "The drive is initialized disabled and motion is rejected until safety limits are set.\n";
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        print_help(argv[0]);
        return 0;
    }
    if (argc != 2) {
        print_help(argv[0]);
        return 2;
    }

    mab::Candle* candle = nullptr;
    mab::MD* md = nullptr;
    bool enabled = false;
    SafetyLimits limits;

    std::signal(SIGINT, request_shutdown);
    std::signal(SIGTERM, request_shutdown);
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    try {
        const int can_id = parse_can_id(argv[1]);
        Gripper::connect(candle, md, can_id, false, false);
        reply_ok("READY " + std::to_string(PROTOCOL_VERSION));

        std::string line;
        while (!shutdown_requested.load() && std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            try {
                std::istringstream input(line);
                std::string command;
                input >> command;

                if (command == "PING") {
                    require_end(input);
                    reply_ok("PONG");
                } else if (command == "CONFIGURE") {
                    configure_limits(input, limits, enabled);
                    reply_ok();
                } else if (command == "CURRENT_LIMIT") {
                    require_configured(limits);
                    if (enabled) {
                        throw std::runtime_error("Disable the drive before changing current limit");
                    }
                    const double value = read_number(input, "current limit");
                    require_end(input);
                    if (value <= 0.0 || value > limits.current_A) {
                        throw std::invalid_argument("Current limit exceeds the configured safety limit");
                    }
                    Gripper::set_current_limit(md, value);
                    limits.current_applied = true;
                    reply_ok();
                } else if (command == "MAX_TORQUE") {
                    require_configured(limits);
                    if (enabled) {
                        throw std::runtime_error("Disable the drive before changing maximum torque");
                    }
                    const double value = read_number(input, "maximum torque");
                    require_end(input);
                    if (value <= 0.0 || value > limits.controller_torque_Nm) {
                        throw std::invalid_argument("Maximum torque exceeds the configured safety limit");
                    }
                    Gripper::set_maximum_torque(md, value);
                    limits.torque_limit_applied = true;
                    reply_ok();
                } else if (command == "ENABLE") {
                    require_end(input);
                    require_configured(limits);
                    if (!limits.current_applied || !limits.torque_limit_applied) {
                        throw std::runtime_error("Apply current and torque limits before enabling");
                    }
                    Gripper::enable(md);
                    enabled = true;
                    reply_ok();
                } else if (command == "DISABLE") {
                    require_end(input);
                    shutdown_drive(md, enabled);
                    reply_ok();
                } else if (command == "CLEAR_FAULTS") {
                    require_end(input);
                    if (enabled) {
                        throw std::runtime_error("Disable the drive before clearing faults");
                    }
                    Gripper::clear_faults(md);
                    reply_ok();
                } else if (command == "ZERO_POSITION") {
                    require_end(input);
                    require_motion_ready(limits, enabled);
                    Gripper::zero_position(md);
                    reply_ok();
                } else if (command == "PROFILE") {
                    require_motion_ready(limits, enabled);
                    const double position = read_number(input, "position");
                    const double velocity = read_number(input, "velocity");
                    const double acceleration = read_number(input, "acceleration");
                    require_end(input);
                    if (position < limits.minimum_position_rad ||
                        position > limits.maximum_position_rad) {
                        throw std::invalid_argument("Position is outside the configured range");
                    }
                    if (velocity <= 0.0 || velocity > limits.maximum_velocity_rad_s ||
                        acceleration <= 0.0 ||
                        acceleration > limits.maximum_acceleration_rad_s2) {
                        throw std::invalid_argument("Profile rate exceeds the configured limit");
                    }
                    Gripper::start_profiled_position(md, position, velocity, acceleration);
                    reply_ok();
                } else if (command == "START_TORQUE") {
                    require_motion_ready(limits, enabled);
                    const double torque = read_number(input, "torque");
                    require_end(input);
                    validate_torque(limits, torque);
                    Gripper::close(md, torque);
                    reply_ok();
                } else if (command == "TORQUE") {
                    require_motion_ready(limits, enabled);
                    const double torque = read_number(input, "torque");
                    require_end(input);
                    validate_torque(limits, torque);
                    Gripper::maintain_close(md, torque);
                    reply_ok();
                } else if (command == "RELEASE_TORQUE") {
                    require_end(input);
                    Gripper::maintain_close(md, 0.0);
                    reply_ok();
                } else if (command == "STOP_IDLE") {
                    require_end(input);
                    Gripper::stop(md);
                    reply_ok();
                } else if (command == "FEEDBACK") {
                    require_end(input);
                    require_configured(limits);
                    const double position = Gripper::read_position(md);
                    const double velocity = Gripper::read_velocity(md);
                    const double torque = Gripper::read_torque(md);
                    const double temperature = Gripper::read_temperature(md);
                    if (temperature > limits.maximum_temperature_C) {
                        throw std::runtime_error("Motor temperature exceeds the configured limit");
                    }
                    std::ostringstream payload;
                    payload << std::setprecision(std::numeric_limits<double>::max_digits10)
                            << position << ' ' << velocity << ' ' << torque << ' ' << temperature;
                    reply_ok(payload.str());
                } else if (command == "QUIT") {
                    require_end(input);
                    shutdown_drive(md, enabled);
                    reply_ok("BYE");
                    break;
                } else {
                    throw std::invalid_argument("Unknown command: " + command);
                }
            } catch (const std::exception& error) {
                // A rejected or failed command must never leave an active motion behind.
                safe_shutdown(md, enabled);
                reply_error(error.what());
            }
        }
    } catch (const std::exception& error) {
        safe_shutdown(md, enabled);
        reply_error(error.what());
        Gripper::disconnect(candle, md);
        return 1;
    }

    safe_shutdown(md, enabled);
    Gripper::disconnect(candle, md);
    return 0;
}

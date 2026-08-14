#include "gripper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
std::atomic_bool keep_running{true};

constexpr double MAX_CURRENT_A = 0.5;
constexpr double MAX_CONTROLLER_TORQUE_NM = 10.0;
constexpr double MIN_POSITION_RAD = 0.0;
constexpr double MAX_POSITION_RAD = 2.2;
constexpr double MAX_VELOCITY_RAD_S = 5.0;
constexpr double MAX_ACCELERATION_RAD_S2 = 20.0;
constexpr double MAX_CLOSING_TORQUE_NM = 4.0;
constexpr double MAX_TEMPERATURE_C = 50.0;
constexpr double TORQUE_RAMP_NM_S = 0.5;
constexpr double MAX_TORQUE_TEST_DURATION_S = 10.0;

extern "C" void request_stop(int) {
    keep_running.store(false);
}

void usage(const char* program) {
    std::cerr
        << "Usage:\n"
        << "  " << program << " probe\n"
        << "  " << program << " enable-test\n"
        << "  " << program << " clear-faults\n"
        << "  " << program << " home [torque_Nm]\n"
        << "  " << program << " position <radians> [velocity_rad_s] [accel_rad_s2]\n"
        << "  " << program << " torque-test <torque_Nm> [duration_s]\n"
        << "  " << program << " close <torque_Nm>\n\n"
        << "probe reads feedback with the drive disabled.\n"
        << "enable-test enables only in IDLE, reads feedback, then immediately disables.\n"
        << "torque-test ramps at 0.5 Nm/s and automatically stops after at most 10 s.\n"
        << "close holds until Ctrl+C, also using the 0.5 Nm/s ramp.\n";
}

double number(const char* text, const char* name) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
    }
    return value;
}

double closing_torque(const char* text) {
    const double value = number(text, "torque");
    if (value >= 0.0 || std::abs(value) > MAX_CLOSING_TORQUE_NM) {
        throw std::invalid_argument("Closing torque must be in [-4, 0) Nm");
    }
    return value;
}

void print_feedback(mab::MD* md) {
    const double position = Gripper::read_position(md);
    const double velocity = Gripper::read_velocity(md);
    const double torque = Gripper::read_torque(md);
    const double temperature = Gripper::read_temperature(md);
    if (temperature > MAX_TEMPERATURE_C) {
        throw std::runtime_error("Motor temperature exceeded 50 C");
    }
    std::cout << std::fixed << std::setprecision(4)
              << "position=" << position << " rad, velocity=" << velocity
              << " rad/s, torque=" << torque << " Nm, temperature="
              << temperature << " C\n";
}

void stop_disable_disconnect(mab::Candle*& candle, mab::MD*& md) noexcept {
    if (md != nullptr) {
        try {
            Gripper::stop(md);
        } catch (...) {
        }
        try {
            Gripper::disable(md);
        } catch (...) {
        }
    }
    Gripper::disconnect(candle, md);
}

void run_torque(
    mab::MD* md,
    double target_torque_Nm,
    std::optional<double> duration_s
) {
    keep_running.store(true);
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);

    Gripper::close(md, 0.0);
    double commanded_torque_Nm = 0.0;
    const auto start = std::chrono::steady_clock::now();
    auto previous = start;
    auto next_report = start;

    while (keep_running.load()) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(now - start).count();
        if (duration_s.has_value() && elapsed_s >= *duration_s) {
            break;
        }

        const double delta_s = std::min(
            std::chrono::duration<double>(now - previous).count(),
            0.1
        );
        previous = now;
        const double maximum_step = TORQUE_RAMP_NM_S * std::max(delta_s, 0.0);
        const double remaining = target_torque_Nm - commanded_torque_Nm;
        if (std::abs(remaining) <= maximum_step) {
            commanded_torque_Nm = target_torque_Nm;
        } else {
            commanded_torque_Nm += std::copysign(maximum_step, remaining);
        }
        Gripper::maintain_close(md, commanded_torque_Nm);

        if (now >= next_report) {
            std::cout << "commanded=" << commanded_torque_Nm << " Nm, ";
            print_feedback(md);
            next_report = now + std::chrono::milliseconds(200);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    Gripper::stop(md);
    std::cout << "Torque released; drive will be disabled.\n";
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    const std::string command = argv[1];
    double torque = 0.0;
    double position = 0.0;
    double velocity = MAX_VELOCITY_RAD_S;
    double acceleration = MAX_ACCELERATION_RAD_S2;
    std::optional<double> torque_duration;

    try {
        if (command == "probe" || command == "enable-test" || command == "clear-faults") {
            if (argc != 2) {
                throw std::invalid_argument(command + " does not accept arguments");
            }
        } else if (command == "home") {
            if (argc > 3) {
                throw std::invalid_argument("home accepts at most one torque argument");
            }
            torque = argc == 3 ? closing_torque(argv[2]) : -0.6;
        } else if (command == "position") {
            if (argc < 3 || argc > 5) {
                throw std::invalid_argument(
                    "position requires radians and optional velocity/acceleration"
                );
            }
            position = number(argv[2], "position");
            velocity = argc >= 4 ? number(argv[3], "velocity") : MAX_VELOCITY_RAD_S;
            acceleration = argc >= 5
                ? number(argv[4], "acceleration")
                : MAX_ACCELERATION_RAD_S2;
            if (position < MIN_POSITION_RAD || position > MAX_POSITION_RAD) {
                throw std::invalid_argument("Position must be in [0, 2.2] rad");
            }
            if (velocity <= 0.0 || velocity > MAX_VELOCITY_RAD_S ||
                acceleration <= 0.0 || acceleration > MAX_ACCELERATION_RAD_S2) {
                throw std::invalid_argument("Profile exceeds 5 rad/s or 20 rad/s^2");
            }
        } else if (command == "torque-test") {
            if (argc < 3 || argc > 4) {
                throw std::invalid_argument(
                    "torque-test requires torque and optional duration"
                );
            }
            torque = closing_torque(argv[2]);
            torque_duration = argc == 4 ? number(argv[3], "duration") : 2.0;
            if (*torque_duration <= 0.0 ||
                *torque_duration > MAX_TORQUE_TEST_DURATION_S) {
                throw std::invalid_argument("Torque-test duration must be in (0, 10] s");
            }
        } else if (command == "close") {
            if (argc != 3) {
                throw std::invalid_argument("close requires one torque argument");
            }
            torque = closing_torque(argv[2]);
        } else {
            usage(argv[0]);
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        usage(argv[0]);
        return 2;
    }

    mab::Candle* candle = nullptr;
    mab::MD* md = nullptr;
    try {
        Gripper::connect(candle, md, Gripper::CAN_ID, false, false);
        Gripper::set_current_limit(md, MAX_CURRENT_A);
        Gripper::set_maximum_torque(md, MAX_CONTROLLER_TORQUE_NM);
        Gripper::stop(md);

        if (command == "probe") {
            std::cout << "Connected to CAN ID " << Gripper::CAN_ID
                      << " with the drive disabled.\n";
            print_feedback(md);
        } else if (command == "clear-faults") {
            Gripper::clear_faults(md);
            std::cout << "Faults cleared while the drive remained disabled.\n";
            print_feedback(md);
        } else {
            Gripper::clear_faults(md);
            Gripper::enable(md);
            if (command == "enable-test") {
                Gripper::stop(md);
                std::cout << "Drive enabled in IDLE; no motion command was sent.\n";
                print_feedback(md);
            } else if (command == "home") {
                Gripper::home(md, torque);
                std::cout << "Homing complete; encoder position was set to zero.\n";
                print_feedback(md);
            } else if (command == "position") {
                Gripper::open_gripper(md, position, velocity, acceleration);
                std::cout << "Position reached.\n";
                print_feedback(md);
            } else if (command == "torque-test") {
                run_torque(md, torque, torque_duration);
            } else if (command == "close") {
                std::cout << "Holding until Ctrl+C.\n";
                run_torque(md, torque, std::nullopt);
            }
        }

        stop_disable_disconnect(candle, md);
        return 0;
    } catch (const std::exception& error) {
        stop_disable_disconnect(candle, md);
        std::cerr << "Motor test failed: " << error.what() << '\n';
        return 1;
    }
}

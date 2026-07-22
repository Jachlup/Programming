#include "gripper.h"

#include "MD.hpp"

#include <chrono>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
std::atomic_bool keep_holding{true};

extern "C" void request_stop(int) {
    keep_holding.store(false);
}

void usage(const char* program) {
    std::cerr
        << "Usage:\n"
        << "  " << program << " home [torque_Nm]\n"
        << "  " << program << " position <radians> [velocity_rad_s] [accel_rad_s2]\n"
        << "  " << program << " close <torque_Nm>\n\n"
        << "Examples:\n"
        << "  " << program << " home -0.6\n"
        << "  " << program << " position 2.0 20 40\n"
        << "  " << program << " close -0.2\n";
}

double number(const char* text, const char* name) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("Invalid ") + name + ": " + text);
    }
    return value;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    mab::Candle* candle = nullptr;
    mab::MD* raw_md = nullptr;

    try {
        Gripper::connect(candle, raw_md);
        mab::MD* md = raw_md;
        const std::string command = argv[1];

        if (command == "home") {
            const double torque = argc >= 3 ? number(argv[2], "torque") : -0.6;
            Gripper::home(md, torque);
            std::cout << "Homing complete; position is zero.\n";
        } else if (command == "position") {
            if (argc < 3) {
                usage(argv[0]);
                return 2;
            }
            const double position = number(argv[2], "position");
            const double velocity = argc >= 4 ? number(argv[3], "velocity") : 20.0;
            const double accel = argc >= 5 ? number(argv[4], "acceleration") : 40.0;
            Gripper::open_gripper(md, position, velocity, accel);
            std::cout << "Position reached.\n";
        } else if (command == "close") {
            if (argc < 3) {
                usage(argv[0]);
                return 2;
            }
            const double torque = number(argv[2], "torque");

            // Keep the process and drive enabled so the torque controller keeps holding.
            // The signal handler only changes a flag; SDK calls remain in normal program flow.
            keep_holding.store(true);
            std::signal(SIGINT, request_stop);
            std::signal(SIGTERM, request_stop);
            Gripper::close(md, torque);
            std::cout << "Holding with " << torque << " Nm. Press Ctrl+C to release.\n";
            auto next_report = std::chrono::steady_clock::now();
            while (keep_holding.load()) {
                Gripper::maintain_close(md, torque);

                const auto now = std::chrono::steady_clock::now();
                if (now >= next_report) {
                    const auto [actual_torque, torque_error] = md->getTorque();
                    const auto [position, position_error] = md->getPosition();
                    if (torque_error != mab::MD::Error_t::OK ||
                        position_error != mab::MD::Error_t::OK) {
                        throw std::runtime_error("Holding feedback read failed");
                    }
                    std::cout << "target=" << torque << " Nm, actual=" << actual_torque
                              << " Nm, position=" << position << " rad\n";
                    next_report = now + std::chrono::milliseconds(500);
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            Gripper::stop(md);
            std::cout << "\nClose torque released.\n";
        } else {
            usage(argv[0]);
            return 2;
        }

        md->disable();
        return 0;
    } catch (const std::exception& error) {
        // Best-effort safe shutdown; do not mask the original failure.
        if (raw_md != nullptr) {
            raw_md->setTargetTorque(0.0f);
            raw_md->setMotionMode(mab::MdMode_E::IDLE);
            raw_md->disable();
        }
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

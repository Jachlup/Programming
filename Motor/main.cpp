#include "gripper.h"
#include <iostream>
#include <thread>
#include <chrono>

// #include <candle/candle.h>
// #include <candle/MD.h>

int main() {
    try {
        // Initialize CANdle and MD
        // CANdle candle;
        // MD md;
        // Gripper::connect(candle, md);
        
        // Home the gripper
        // std::cout << "Homing gripper..." << std::endl;
        // Gripper::home(md);
        
        // Wait for stabilization
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Open the gripper
        // std::cout << "Opening gripper..." << std::endl;
        // Gripper::open_gripper(md);
        
        // Wait
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Set max torque
        // md.setMaxTorque(2.0);
        
        // Apply small torque
        // md.setTargetTorque(0.2);
        
        // Wait
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Release torque
        // md.setTargetTorque(0.0);
        
        std::cout << "Gripper control example (stubs - integrate with CANdle SDK)" << std::endl;
        std::cout << "Uncomment code and link CANdle SDK to compile." << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
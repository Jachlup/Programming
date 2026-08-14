cmake_minimum_required(VERSION 3.16)
project(GripperControl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# CANdle SDK Configuration
# ============================================================================

# Path to CANdle SDK (adjust if needed)
set(CANDLE_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Repos/CANdle-SDK" CACHE PATH "Path to CANdle SDK")

if(NOT EXISTS "${CANDLE_SDK_PATH}")
    message(FATAL_ERROR "CANdle SDK not found at ${CANDLE_SDK_PATH}")
endif()

# Try to find CANdle via CMake first
find_package(Candle QUIET PATHS "${CANDLE_SDK_PATH}/cmake")

if(NOT Candle_FOUND)
    message(STATUS "CANdle package not found, building from source...")
    
    # Define the two commons targets needed by candlelib. We do this directly
    # because the SDK's full commons project also requires optional git submodules.
    add_subdirectory("${CANDLE_SDK_PATH}/commons/shared_data" shared_data EXCLUDE_FROM_ALL)
    add_library(logger STATIC "${CANDLE_SDK_PATH}/commons/logger/logger.cpp")
    target_include_directories(logger PUBLIC "${CANDLE_SDK_PATH}/commons/logger")
    set_target_properties(logger PROPERTIES POSITION_INDEPENDENT_CODE ON)
    
    # Now add candlelib
    if(EXISTS "${CANDLE_SDK_PATH}/candlelib/CMakeLists.txt")
        add_subdirectory("${CANDLE_SDK_PATH}/candlelib" candlelib EXCLUDE_FROM_ALL)
        message(STATUS "Added candlelib subdirectory")
    else()
        message(FATAL_ERROR "candlelib/CMakeLists.txt not found")
    endif()
    
    # Set variables for later use - link to 'candle' target, not 'candlelib'
    set(CANDLE_TARGET candle)
    set(CANDLE_INCLUDE_DIRS 
        "${CANDLE_SDK_PATH}/candlelib/include"
    )
else()
    message(STATUS "Found CANdle SDK")
    set(CANDLE_TARGET candle)
endif()

# ============================================================================
# Gripper Control Library
# ============================================================================

add_library(gripper_lib STATIC
    gripper.cpp
)

target_include_directories(gripper_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CANDLE_INCLUDE_DIRS}
)

target_link_libraries(gripper_lib PUBLIC
    ${CANDLE_TARGET}
)

# ============================================================================
# Main Executable
# ============================================================================

add_executable(gripper_control
    main.cpp
)

add_executable(gripper_bridge
    motor_bridge.cpp
)

target_link_libraries(gripper_control PRIVATE
    gripper_lib
    ${CANDLE_TARGET}
)

target_include_directories(gripper_control PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CANDLE_INCLUDE_DIRS}
)

target_link_libraries(gripper_bridge PRIVATE
    gripper_lib
    ${CANDLE_TARGET}
)

target_include_directories(gripper_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CANDLE_INCLUDE_DIRS}
)

# ============================================================================
# Compiler Flags
# ============================================================================

if(UNIX)
    target_compile_options(gripper_control PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(gripper_bridge PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(gripper_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MSVC)
    target_compile_options(gripper_control PRIVATE /W4)
    target_compile_options(gripper_bridge PRIVATE /W4)
    target_compile_options(gripper_lib PRIVATE /W4)
endif()

# ============================================================================
# Build Options
# ============================================================================

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_DOCS "Build documentation" OFF)

message(STATUS "=================================================")
message(STATUS "Gripper Control Configuration")
message(STATUS "=================================================")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CANDLE_SDK_PATH: ${CANDLE_SDK_PATH}")
message(STATUS "CANDLE_INCLUDE_DIRS: ${CANDLE_INCLUDE_DIRS}")
message(STATUS "CANDLE_TARGET: ${CANDLE_TARGET}")
message(STATUS "=================================================")

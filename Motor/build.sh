#!/bin/bash
# Build script for Gripper Control

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"

echo "=================================="
echo "Gripper Control Build Script"
echo "=================================="
echo "Source directory: ${SCRIPT_DIR}"
echo "Build directory: ${BUILD_DIR}"
echo ""

# Create build directory
if [ ! -d "${BUILD_DIR}" ]; then
    mkdir -p "${BUILD_DIR}"
    echo "Created build directory"
fi

# Configure with CMake
echo "Configuring CMake..."
cd "${BUILD_DIR}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo ""
echo "Building..."
cmake --build . --config Release --parallel $(nproc || echo 4)

echo ""
echo "=================================="
echo "Build complete!"
echo "=================================="
echo "Executable: ${BUILD_DIR}/gripper_control"
echo "Library: ${BUILD_DIR}/libgripper_lib.a"
echo ""
echo "To run: ${BUILD_DIR}/gripper_control"
echo "To install: cmake --install . --prefix ${INSTALL_DIR}"

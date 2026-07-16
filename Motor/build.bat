@echo off
REM Build script for Gripper Control (Windows)

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "INSTALL_DIR=%SCRIPT_DIR%install"

echo ==================================
echo Gripper Control Build Script (Windows)
echo ==================================
echo Source directory: %SCRIPT_DIR%
echo Build directory: %BUILD_DIR%
echo.

REM Create build directory
if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
    echo Created build directory
)

REM Configure with CMake
echo Configuring CMake...
cd /d "%BUILD_DIR%"
cmake .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

REM Build
echo.
echo Building...
cmake --build . --config Release --parallel 4

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo ==================================
echo Build complete!
echo ==================================
echo Executable: %BUILD_DIR%\Release\gripper_control.exe
echo Library: %BUILD_DIR%\gripper_lib.lib
echo.
echo To run: %BUILD_DIR%\Release\gripper_control.exe
echo To install: cmake --install . --prefix %INSTALL_DIR%

endlocal

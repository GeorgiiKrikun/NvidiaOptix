# NVIDIA OptiX 7 Minimal Working Example (MWE)

This project provides a minimal example of an NVIDIA OptiX 7 application. It renders a single triangle and saves the output as a PPM image.

## Project Structure

- `include/common.h`: Shared data structures between host and device.
- `src/device_programs.cu`: OptiX device programs (RayGen, Miss, ClosestHit).
- `src/main.cpp`: Host code for OptiX initialization, resource management, and launch.
- `CMakeLists.txt`: Build configuration.
- `run_tests.sh`: Structural validation script.

## Requirements

- **NVIDIA GPU** with Maxwell architecture or newer.
- **CUDA Toolkit** (10.0 or newer recommended).
- **NVIDIA OptiX SDK** (7.0 or newer).
- **CMake** (3.12 or newer).

## Building and Running

1. Set the `OptiX_INSTALL_DIR` to your OptiX SDK location.
2. Use CMake to configure and build:

```bash
mkdir build
cd build
cmake -DOptiX_INSTALL_DIR=/path/to/your/OptiX/SDK ..
make
```

3. Run the application:

```bash
./optix_mwe
```

The output will be saved to `output.ppm`.

## Note on this Example

This code is based on the NVIDIA OptiX 7 Quickstart guide and follows modern OptiX best practices, including explicit memory management and the use of the Shader Binding Table (SBT).

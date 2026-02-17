#!/bin/bash

echo "Starting structural validation of OptiX MWE..."

# Check for nvcc
if command -v nvcc &> /dev/null
then
    echo "✅ nvcc found: $(nvcc --version | head -n 1)"
else
    echo "❌ nvcc NOT found. CUDA is required to build this project."
fi

# Check for CMake
if command -v cmake &> /dev/null
then
    echo "✅ cmake found: $(cmake --version | head -n 1)"
else
    echo "❌ cmake NOT found."
fi

echo "Attempting to configure project with CMake..."
mkdir -p build
cd build
cmake ..

if [ $? -eq 0 ]; then
    echo "✅ CMake configuration successful."
    echo "Note: Full compilation will likely fail if OptiX SDK or CUDA is missing."
else
    echo "❌ CMake configuration failed. This is expected if CUDA or OptiX is not installed."
fi

echo "Structural validation complete."

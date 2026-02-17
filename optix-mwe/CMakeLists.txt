cmake_minimum_required(VERSION 3.17)
project(optix_mwe LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# User should set OptiX_INSTALL_DIR to the location of the OptiX SDK
set(OptiX_INSTALL_DIR "" CACHE PATH "Path to OptiX SDK")

# Try to find optix.h
find_path(OptiX_INCLUDE_DIR optix.h
    HINTS "${OptiX_INSTALL_DIR}"
    PATH_SUFFIXES include
)

if(NOT OptiX_INCLUDE_DIR)
    message(FATAL_ERROR
        "OptiX headers not found! Please set OptiX_INSTALL_DIR to your OptiX SDK location "
        "or set OptiX_INCLUDE_DIR directly to the directory containing optix.h."
    )
endif()

message(STATUS "Found OptiX headers: ${OptiX_INCLUDE_DIR}")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OptiX_INCLUDE_DIR})

# Manual PTX generation using the CUDA compiler identified by CMake
set(PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/device_programs.ptx")
add_custom_command(
    OUTPUT "${PTX_FILE}"
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/device_programs.cu"
            -o "${PTX_FILE}"
            -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I"${OptiX_INCLUDE_DIR}"
            -arch=sm_60
    DEPENDS src/device_programs.cu include/common.h
    VERBATIM
)
add_custom_target(generate_ptx ALL DEPENDS "${PTX_FILE}")

# Host code
add_executable(optix_mwe src/main.cpp)
add_dependencies(optix_mwe generate_ptx)

# Use CUDAToolkit for modern CUDA library linking
find_package(CUDAToolkit REQUIRED)
target_link_libraries(optix_mwe PRIVATE CUDA::cudart)

# Link against CUDA driver library (libcuda.so)
if(TARGET CUDA::cuda_driver)
    target_link_libraries(optix_mwe PRIVATE CUDA::cuda_driver)
elseif(TARGET CUDA::cuda)
    target_link_libraries(optix_mwe PRIVATE CUDA::cuda)
else()
    # Fallback to finding it manually
    find_library(CUDA_DRIVER_LIB cuda)
    if(CUDA_DRIVER_LIB)
        target_link_libraries(optix_mwe PRIVATE ${CUDA_DRIVER_LIB})
    endif()
endif()

# Define the absolute path to the PTX file for the host code
target_compile_definitions(optix_mwe PRIVATE OPTIX_PTX_PATH="${PTX_FILE}")

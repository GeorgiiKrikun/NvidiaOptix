cmake_minimum_required(VERSION 3.12)
project(optix_mwe LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# User should set OptiX_INSTALL_DIR to the location of the OptiX SDK
set(OptiX_INSTALL_DIR "" CACHE PATH "Path to OptiX SDK")

if(NOT OptiX_INSTALL_DIR)
    message(WARNING "OptiX_INSTALL_DIR not set. Compilation might fail.")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OptiX_INSTALL_DIR}/include
)

# Device code
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60 -ptx")
cuda_add_library(device_programs PTX src/device_programs.cu)

# Host code
add_executable(optix_mwe src/main.cpp)
target_link_libraries(optix_mwe ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

# Note: In OptiX 7, you often need to include optix_stubs.h and link against cuda
# The optix_stubs.h is usually included in the main.cpp as I did.

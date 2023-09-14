
cmake_minimum_required(VERSION 3.5)

# CUDA settings
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(fusion-test VERSION 1.0.0
                    LANGUAGES CXX CUDA)

# ------------------------------------------------------------------
# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------
# Build
add_library(${CMAKE_PROJECT_NAME} SHARED kernels.cu)

# Set Properties
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cuda_std_14)
set_target_properties(${CMAKE_PROJECT_NAME}
PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared
  # CUDA_STANDARD 14
  # CUDA_SEPARABLE_COMPILATION ON
)
cmake_minimum_required(VERSION 3.5)
project(bfuse VERSION 1.0.0)
# ------------------------------------------------------------------
# Configure files
configure_file(BFuseConfig.h.in BFuseConfig.h)

# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-offsetof")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# ------------------------------------------------------------------
# LLVM & Clang path
set(LLVM_DIR "/usr/lib/llvm-16/cmake")
set(Clang_DIR "/usr/lib/cmake/clang-16")

# Find LLVM & Clang
find_package(LLVM CONFIG REQUIRED HINTS ${LLVM_DIR})
message(STATUS "Using LLVM version ${LLVM_PACKAGE_VERSION}")
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

find_package(Clang CONFIG REQUIRED HINTS ${Clang_DIR})
list (APPEND CMAKE_MODULE_PATH ${CLANG_CMAKE_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

include_directories(${CLANG_INCLUDE_DIRS})

include(AddLLVM)
include(HandleLLVMOptions)
include(AddClang)
# ------------------------------------------------------------------
# Build
add_executable(${CMAKE_PROJECT_NAME} main.cc)
# ------------------------------------------------------------------
# BFuse
add_subdirectory(bfuse)
list(APPEND EXTRA_LIBS bfuse)
# ------------------------------------------------------------------
# HFuse
add_subdirectory(hfuse)
list(APPEND EXTRA_LIBS hfuse)
# ------------------------------------------------------------------
# Link libraries
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${EXTRA_LIBS})

# Include configure files
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_BINARY_DIR})
# ------------------------------------------------------------------
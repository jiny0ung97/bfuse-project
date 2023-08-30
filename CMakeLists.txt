
cmake_minimum_required(VERSION 3.5)

project(bfuseplus VERSION 1.0.0)

# ------------------------------------------------------------------
# Configure files
configure_file(BfuseConfig.h.in BfuseConfig.h)

# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-offsetof")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ------------------------------------------------------------------
# Find LLVM
find_package(LLVM
CONFIG
  REQUIRED
  PATHS    "/usr/lib/llvm-10/cmake"
)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

include(HandleLLVMOptions)
include(AddLLVM)

# Find Clang
find_package(Clang
CONFIG
  REQUIRED
  PATHS    "/usr/lib/cmake/clang-10"
)
list (APPEND CMAKE_MODULE_PATH ${CLANG_CMAKE_DIR})

include_directories(${CLANG_INCLUDE_DIRS})

# ------------------------------------------------------------------
# Options for libraries
option(USE_BFUSE "Use the BFuse library" ON)

# BFuse project library
if(USE_BFUSE)
  add_subdirectory(bfuse)
  list(APPEND EXTRA_LIBS bfuse)
endif()

# ------------------------------------------------------------------
# Build & Link
add_executable(${CMAKE_PROJECT_NAME} main.cc)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${EXTRA_LIBS})

# Include configure files
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_BINARY_DIR})
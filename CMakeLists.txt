
cmake_minimum_required(VERSION 3.5)

project(bfuseplus VERSION 1.0.0)

# ------------------------------------------------------------------
# Configure files
configure_file(BfuseConfig.h.in BfuseConfig.h)

# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-offsetof")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ------------------------------------------------------------------
# Find Clang
find_package(Clang CONFIG REQUIRED)
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
# Options for libraries
option(USE_BFUSE "Use the bfuse library" ON)
option(USE_TEST "Use the test library" ON)

# BFuse project library
if(USE_BFUSE)
  add_subdirectory(bfuse)
  list(APPEND EXTRA_LIBS bfuse)
  target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE "USE_BFUSE")
endif()

# Test
if(USE_TEST)
  add_subdirectory(test)
endif()

# ------------------------------------------------------------------
# Link libraries
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${EXTRA_LIBS})

# Include configure files
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_BINARY_DIR})
# ------------------------------------------------------------------
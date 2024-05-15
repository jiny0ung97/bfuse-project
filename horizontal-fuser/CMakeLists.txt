cmake_minimum_required(VERSION 3.5)
project(horizontal-fuser VERSION 1.0.0)
# ------------------------------------------------------------------
# Configure files
configure_file(CMakeConfig.h.in CMakeConfig.h)

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
# Sources
set(FUSER_SOURCE_DIR src)
set(FUSER_SOURCES
  ${FUSER_SOURCE_DIR}/main.cc
  ${FUSER_SOURCE_DIR}/Fuse.cc
  ${FUSER_SOURCE_DIR}/Utils.cc
  ${FUSER_SOURCE_DIR}/Contexts.cc
  ${FUSER_SOURCE_DIR}/Tools.cc
  ${FUSER_SOURCE_DIR}/Matchers.cc
  # Add your sources here
  # ${FUSER_SOURCE_DIR}/foo/bar/your_source.cc
)

# Headers
set(FUSER_HEADER_DIR include)
set(FUSER_HEADERS
  ${FUSER_HEADER_DIR}/fuse/Fuse.h
  ${FUSER_HEADER_DIR}/fuse/Utils.h
  ${FUSER_HEADER_DIR}/fuse/Contexts.h
  ${FUSER_HEADER_DIR}/fuse/Tools.h
  ${FUSER_HEADER_DIR}/fuse/Matchers.h
  # Add your headers here
  # ${FUSER_HEADER_DIR}/foo/bar/your_header.h
)
# ------------------------------------------------------------------
# Build
set(LLVM_LINK_COMPONENTS support)
add_clang_executable(${CMAKE_PROJECT_NAME} ${FUSER_HEADERS} ${FUSER_SOURCES})

target_include_directories(${CMAKE_PROJECT_NAME}
PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/${FUSER_HEADER_DIR}
)

# Link against Clang libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  clangToolingRefactoring
)
# ------------------------------------------------------------------
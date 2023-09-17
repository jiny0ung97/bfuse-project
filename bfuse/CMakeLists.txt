
# Sources
set(BFUSE_SOURCE_DIR src)
set(BFUSE_SOURCES
  ${BFUSE_SOURCE_DIR}/Bfuse.cc
  ${BFUSE_SOURCE_DIR}/Utils.cc
  ${BFUSE_SOURCE_DIR}/Contexts.cc
  ${BFUSE_SOURCE_DIR}/Instances.cc
  ${BFUSE_SOURCE_DIR}/Matchers.cc
  # ${BFUSE_SOURCE_DIR}/Instances.cc
  # Add your sources here
  # ${BFUSE_SOURCE_DIR}/foo/bar/your_source.cc
)

# Headers
set(BFUSE_HEADER_DIR include)
set(BFUSE_HEADERS
  ${BFUSE_HEADER_DIR}/bfuse/Bfuse.h
  ${BFUSE_HEADER_DIR}/bfuse/Utils.h
  ${BFUSE_HEADER_DIR}/bfuse/Contexts.h
  ${BFUSE_HEADER_DIR}/bfuse/Instances.h
  ${BFUSE_HEADER_DIR}/bfuse/Matchers.h
  # ${BFUSE_HEADER_DIR}/bfuse/Instances.h
  # Add your headers here
  # ${BFUSE_HEADER_DIR}/foo/bar/your_header.h
)

# ------------------------------------------------------------------
# Link with add_clang_library function()
set(LLVM_LINK_COMPONENTS support)
add_clang_library(bfuse STATIC ${BFUSE_HEADERS} ${BFUSE_SOURCES})

target_include_directories(bfuse
PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/${BFUSE_HEADER_DIR}
)

# Link against Clang libraries
target_link_libraries(bfuse
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
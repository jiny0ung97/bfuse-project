
# Sources
set(BFUSE_SOURCE_DIR src)
set(BFUSE_SOURCES
  ${BFUSE_SOURCE_DIR}/bfuse.cc
  ${BFUSE_SOURCE_DIR}/utils.cc
  # Add your sources here
  # ${BFUSE_SOURCE_DIR}/foo/bar/your_source.cc
)

# Headers
set(BFUSE_HEADER_DIR include)
set(BFUSE_HEADERS
  ${BFUSE_HEADER_DIR}/bfuse.h
  ${BFUSE_HEADER_DIR}/utils.h
  # Add your headers here
  # ${BFUSE_HEADER_DIR}/foo/bar/your_header.h
)

add_library(bfuse STATIC ${BFUSE_HEADERS} ${BFUSE_SOURCES})

target_include_directories(bfuse
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${BFUSE_HEADER_DIR}
)

# Find the libraries that correspond to the LLVM components
# that we wish to use
# llvm_map_components_to_libnames(llvm_libs support core irreader passes)

# Link against LLVM libraries
# target_link_libraries(bfuse PRIVATE ${llvm_libs})

# Link against Clang libraries
set(LLVM_LINK_COMPONENTS support)
target_link_libraries(bfuse
PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
)
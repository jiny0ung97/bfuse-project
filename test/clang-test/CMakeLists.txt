
# Clang-test sources
set(BFUSE_CLANG_TEST_SOURCE main.cc)

# ------------------------------------------------------------------
# Link with add_clang_library function()
set(LLVM_LINK_COMPONENTS support)
add_clang_executable(clang-test ${BFUSE_CLANG_TEST_SOURCE})

# Link against Clang libraries
target_link_libraries(clang-test
PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
)
# ------------------------------------------------------------------
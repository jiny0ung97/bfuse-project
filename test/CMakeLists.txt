
# Options for tests
option(USE_FUSION_TEST "Use fusion-test" OFF)
option(USE_CLANG_TEST "Use clang test" OFF)

# Fusion test
if(USE_FUSION_TEST)
  add_subdirectory(fusion-test)
endif()

# Clang test
if(USE_CLANG_TEST)
  add_subdirectory(clang-test)
endif()
# ------------------------------------------------------------------
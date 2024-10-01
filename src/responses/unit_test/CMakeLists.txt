configure_file( VertexCFD_ResponseUnitTestConfig.hpp.cmakein VertexCFD_ResponseUnitTestConfig.hpp )

set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

VertexCFD_add_tests(
  LIBS VertexCFD
  NAMES
  ResponseManager
  ResponseUtils
  )

# The ResponseManager test relies on Panzer capability that uses CUDA UVM.
# This must be launched with CUDA_LAUNCH_BLOCKING to work correctly.
if(${VERTEXCFD_KOKKOS_DEVICE_TYPE} STREQUAL "CUDA")
  message("Setting ResponseManager launch env")
  set_property(TEST VertexCFD_ResponseManager_test_CUDA PROPERTY ENVIRONMENT "CUDA_LAUNCH_BLOCKING=1")
endif()

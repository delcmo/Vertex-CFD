configure_file( VertexCFD_DriverUnitTestConfig.hpp.cmakein VertexCFD_DriverUnitTestConfig.hpp )

set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

VertexCFD_add_tests(
  LIBS VertexCFD
  NAMES
  MeshManager
  PhysicsManager
  InitialConditionManager
  )

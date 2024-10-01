set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_data/read_only_test.restart.data
  ${CMAKE_CURRENT_BINARY_DIR}/read_only_test.restart.data
  COPYONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_data/read_only_test.restart.dofmap
  ${CMAKE_CURRENT_BINARY_DIR}/read_only_test.restart.dofmap
  COPYONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_data/read_only_test_periodic.restart.data
  ${CMAKE_CURRENT_BINARY_DIR}/read_only_test_periodic.restart.data
  COPYONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_data/read_only_test_periodic.restart.dofmap
  ${CMAKE_CURRENT_BINARY_DIR}/read_only_test_periodic.restart.dofmap
  COPYONLY)

VertexCFD_add_tests(
  MPI
  LIBS VertexCFD
  NAMES Restart GeometryPrimitives
  )

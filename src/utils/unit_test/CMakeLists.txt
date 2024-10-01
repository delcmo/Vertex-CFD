set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

VertexCFD_add_tests(
  LIBS Utils ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
  NAMES 
  Version
  ParameterPack
  SmoothMath
  Constants
  TypeTraits
  MatrixMath
  NonlinearSolver
  ScalarToVector
  VectorizeOutputFieldNames 
  )

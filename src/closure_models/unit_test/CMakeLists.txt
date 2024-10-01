set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)
include(${TEST_HARNESS_DIR}/TestHarness.cmake)

VertexCFD_add_tests(
  LIBS VertexCFD
  NAMES
  ClosureModelFactoryTestHarness
  ElementLength
  MetricTensor
  MetricTensorElementLength
  MethodManufacturedSolution
  MethodManufacturedSolutionSource
  MeasureElementLength
  SingularValueElementLength
  WallDistance
  VectorFieldDivergence
  ConstantScalarField
  )

VertexCFD_add_tests(
  MPI
  LIBS VertexCFD
  NAMES
  ExternalFields
  )

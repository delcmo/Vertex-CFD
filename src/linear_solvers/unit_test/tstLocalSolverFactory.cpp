#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <linear_solvers/VertexCFD_LinearSolvers_CusolverGLU.hpp>
#include <linear_solvers/VertexCFD_LinearSolvers_LocalSolverFactory.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
TEST(LocalSolverFactoryTester, build_test)
{
    // Test with default parameters
    Teuchos::ParameterList params;
    auto solver
        = VertexCFD::LinearSolvers::LocalSolverFactory::buildSolver(params);
    auto cusolver
        = std::dynamic_pointer_cast<VertexCFD::LinearSolvers::CusolverGLU>(
            solver);
    EXPECT_TRUE(cusolver != nullptr);

    // Test with solver name
    params.set("Local Solver", "Cusolver GLU");
    solver = VertexCFD::LinearSolvers::LocalSolverFactory::buildSolver(params);
    cusolver = std::dynamic_pointer_cast<VertexCFD::LinearSolvers::CusolverGLU>(
        solver);
    EXPECT_TRUE(cusolver != nullptr);

    // Test with invalid name
    params.set("Local Solver", "Foo");
    EXPECT_THROW(
        VertexCFD::LinearSolvers::LocalSolverFactory::buildSolver(params),
        std::runtime_error);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD

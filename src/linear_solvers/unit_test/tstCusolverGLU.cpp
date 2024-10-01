#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <VertexCFD_SolverTester.hpp>
#include <linear_solvers/VertexCFD_LinearSolvers_CusolverGLU.hpp>

#include <Teuchos_DefaultMpiComm.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
class CusolverGLUTester : public SolverTester
{
  protected:
    void solve() const
    {
        // Build and compute preconditioner
        Teuchos::ParameterList params;
        VertexCFD::LinearSolvers::CusolverGLU solver(params);
        solver.setMatrix(_matrix);
        solver.initialize();
        solver.compute();

        // In operator notation, the input vector "x" becomes the RHS for a
        // linear solve and the output "y" is the result
        solver.solve(*_x, *_y);
    }
};

//---------------------------------------------------------------------------//
TEST_F(CusolverGLUTester, solve_test)
{
    this->solve();

    auto y_data = _y->getData(0);
    int num_local_rows = y_data.size();
    double tol = 1e-14;
    for (int local_row = 0; local_row < num_local_rows; ++local_row)
    {
        EXPECT_NEAR(_ref_soln[local_row], y_data[local_row], tol);
    }
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD

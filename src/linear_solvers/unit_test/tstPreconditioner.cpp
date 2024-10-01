#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <VertexCFD_SolverTester.hpp>
#include <linear_solvers/VertexCFD_LinearSolvers_Preconditioner.hpp>

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
class PreconditionerTester : public SolverTester
{
};

//---------------------------------------------------------------------------//
TEST_F(PreconditionerTester, solve_test)
{
    // Build and compute preconditioner
    Teuchos::ParameterList params;
    VertexCFD::LinearSolvers::Preconditioner prec;
    prec.setParameters(params);
    EXPECT_FALSE(prec.isInitialized());
    EXPECT_FALSE(prec.isComputed());
    EXPECT_EQ(0, prec.getNumInitialize());
    EXPECT_EQ(0, prec.getNumCompute());
    EXPECT_EQ(0, prec.getNumApply());

    // Set matrix
    prec.setMatrix(_matrix);
    EXPECT_FALSE(prec.isInitialized());
    EXPECT_FALSE(prec.isComputed());
    EXPECT_EQ(0, prec.getNumInitialize());
    EXPECT_EQ(0, prec.getNumCompute());
    EXPECT_EQ(0, prec.getNumApply());

    // Initialize preconditioner
    prec.initialize();
    EXPECT_TRUE(prec.isInitialized());
    EXPECT_FALSE(prec.isComputed());
    EXPECT_EQ(1, prec.getNumInitialize());
    EXPECT_EQ(0, prec.getNumCompute());
    EXPECT_EQ(0, prec.getNumApply());

    // Compute preconditioner
    prec.compute();
    EXPECT_TRUE(prec.isInitialized());
    EXPECT_TRUE(prec.isComputed());
    EXPECT_EQ(1, prec.getNumInitialize());
    EXPECT_EQ(1, prec.getNumCompute());
    EXPECT_EQ(0, prec.getNumApply());

    // Apply preconditioner with beta = 0.
    // Existing values in y should be ignored
    _x->putScalar(1.0);
    _y->putScalar(2.0);
    prec.apply(*_x, *_y, Teuchos::NO_TRANS, 1.0, 0.0);
    EXPECT_TRUE(prec.isInitialized());
    EXPECT_TRUE(prec.isComputed());
    EXPECT_EQ(1, prec.getNumInitialize());
    EXPECT_EQ(1, prec.getNumCompute());
    EXPECT_EQ(1, prec.getNumApply());

    {
        auto y_data = _y->getData(0);
        int num_local_rows = y_data.size();
        double tol = 1e-14;
        for (int local_row = 0; local_row < num_local_rows; ++local_row)
        {
            EXPECT_NEAR(_ref_soln[local_row], y_data[local_row], tol);
        }
    }

    // Apply preconditioner with alpha = 0.
    // Values in x should be ignored. No solve is performed, just scaling on y.
    _x->putScalar(1.0);
    _y->putScalar(2.0);
    prec.apply(*_x, *_y, Teuchos::NO_TRANS, 0.0, 1.5);
    EXPECT_TRUE(prec.isInitialized());
    EXPECT_TRUE(prec.isComputed());
    EXPECT_EQ(1, prec.getNumInitialize());
    EXPECT_EQ(1, prec.getNumCompute());
    EXPECT_EQ(2, prec.getNumApply());

    {
        auto y_data = _y->getData(0);
        int num_local_rows = y_data.size();
        for (int local_row = 0; local_row < num_local_rows; ++local_row)
        {
            EXPECT_DOUBLE_EQ(3.0, y_data[local_row]);
        }
    }

    // Apply preconditioner with nonzero alpha and beta.
    // Values in both x and y are used.
    _x->putScalar(1.0);
    _y->putScalar(2.0);
    prec.apply(*_x, *_y, Teuchos::NO_TRANS, 2.0, 3.0);
    EXPECT_TRUE(prec.isInitialized());
    EXPECT_TRUE(prec.isComputed());
    EXPECT_EQ(1, prec.getNumInitialize());
    EXPECT_EQ(1, prec.getNumCompute());
    EXPECT_EQ(3, prec.getNumApply());

    {
        auto y_data = _y->getData(0);
        int num_local_rows = y_data.size();
        double tol = 2e-14;
        for (int local_row = 0; local_row < num_local_rows; ++local_row)
        {
            EXPECT_NEAR(
                2.0 * _ref_soln[local_row] + 6.0, y_data[local_row], tol);
        }
    }
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD

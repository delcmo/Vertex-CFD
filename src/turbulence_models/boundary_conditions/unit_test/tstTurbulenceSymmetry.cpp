#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceSymmetry.hpp"
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

#include <mpi.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _variable;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    Dependencies(const panzer::IntegrationRule& ir)
        : _variable("variable", ir.dl_scalar)
        , _grad_variable("GRAD_variable", ir.dl_vector)
        , _normals("Side Normal", ir.dl_vector)
    {
        this->addEvaluatedField(_variable);
        this->addEvaluatedField(_grad_variable);
        this->addEvaluatedField(_normals);
        this->setName("Turbulence Model Symmetry Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "turbulence symmetry test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        using std::sqrt;

        const int num_point = _variable.extent(1);
        const int num_space_dim = _normals.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            _variable(c, qp) = 2.0;

            for (int d = 0; d < num_space_dim; ++d)
            {
                _grad_variable(c, qp, d) = 0.02 * (d + 1.0);
                _normals(c, qp, d) = 0.33 * (d + 1.0);
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_grad_dim)
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create symmetry evaluator.
    const std::string variable_name = "variable";
    const auto symm_eval = Teuchos::rcp(
        new BoundaryCondition::TurbulenceSymmetry<EvalType, panzer::Traits>(
            *test_fixture.ir, variable_name));
    test_fixture.registerEvaluator<EvalType>(symm_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(symm_eval->_boundary_variable);
    test_fixture.registerTestField<EvalType>(
        symm_eval->_boundary_grad_variable);

    // Evaluate values
    test_fixture.evaluate<EvalType>();

    // Check values
    const auto boundary_var_result = test_fixture.getTestFieldData<EvalType>(
        symm_eval->_boundary_variable);
    const auto boundary_grad_var_result
        = test_fixture.getTestFieldData<EvalType>(
            symm_eval->_boundary_grad_variable);

    const int num_point = boundary_var_result.extent(1);

    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    const double exp_grad_2D[3] = {0.00911, 0.01822, nan_val};
    const double exp_grad_3D[3] = {
        -0.010492000000000001, -0.020984000000000003, -0.031476000000000004};
    const auto exp_grad_value = (num_grad_dim == 3) ? exp_grad_3D : exp_grad_2D;

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(2.0, fieldValue(boundary_var_result, 0, qp));

        for (int d = 0; d < num_grad_dim; ++d)
        {
            EXPECT_DOUBLE_EQ(exp_grad_value[d],
                             fieldValue(boundary_grad_var_result, 0, qp, d));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D case
TEST(Test2DTurbulenceSymmetry, residual)
{
    testEval<panzer::Traits::Residual>(2);
}

TEST(Test2DTurbulenceSymmetry, jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//---------------------------------------------------------------------------//
// 3-D case
TEST(Test3DTurbulenceSymmetry, residual)
{
    testEval<panzer::Traits::Residual>(3);
}

TEST(Test3DTurbulenceSymmetry, jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD

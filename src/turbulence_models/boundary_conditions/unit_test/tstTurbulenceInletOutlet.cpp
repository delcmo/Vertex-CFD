#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceInletOutlet.hpp"
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <utils/VertexCFD_Utils_VectorField.hpp>

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

    double _u0, _u1, _u2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _variable;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u0,
                 const double u1,
                 const double u2)
        : _u0(u0)
        , _u1(u1)
        , _u2(u2)
        , _variable("variable", ir.dl_scalar)
        , _grad_variable("GRAD_variable", ir.dl_vector)
        , _velocity_0("velocity_0", ir.dl_scalar)
        , _velocity_1("velocity_1", ir.dl_scalar)
        , _velocity_2("velocity_2", ir.dl_scalar)
        , _normals("Side Normal", ir.dl_vector)
    {
        this->addEvaluatedField(_variable);
        this->addEvaluatedField(_grad_variable);
        this->addEvaluatedField(_velocity_0);
        this->addEvaluatedField(_velocity_1);
        this->addEvaluatedField(_velocity_2);
        this->addEvaluatedField(_normals);
        this->setName("Turbulence Model Inlet/Outlet Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        _variable.deep_copy(2.0);
        _grad_variable.deep_copy(3.0);
        _normals.deep_copy(4.0);
        _velocity_0.deep_copy(_u0);
        _velocity_1.deep_copy(_u1);
        _velocity_2.deep_copy(_u2);
    }
};

//---------------------------------------------------------------------------//
// Inlet/outlet cases
enum class InletOutlet
{
    inlet,
    outlet
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const InletOutlet inlet_outlet)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Set sign for inlet or outlet flow
    double sign = 0.0;

    switch (inlet_outlet)
    {
        case (InletOutlet::inlet):
            sign = -1.0;
            break;
        case (InletOutlet::outlet):
            sign = 1.0;
            break;
    }

    // Create dependencies
    const double _nanval = std::numeric_limits<double>::quiet_NaN();
    const double u0 = 5.0 * sign;
    const double u1 = 6.0 * sign;
    const double u2 = num_space_dim == 3 ? 7.0 * sign : _nanval;

    const auto dep_eval = Teuchos::rcp(
        new Dependencies<EvalType>(*test_fixture.ir, u0, u1, u2));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create inlet/outlet evaluator.
    const std::string variable_name = "variable";
    Teuchos::ParameterList bc_params;
    bc_params.set("variable Inlet Value", 2.75);
    const auto fixed_eval = Teuchos::rcp(
        new BoundaryCondition::
            TurbulenceInletOutlet<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir, bc_params, variable_name));
    test_fixture.registerEvaluator<EvalType>(fixed_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(fixed_eval->_boundary_variable);
    test_fixture.registerTestField<EvalType>(
        fixed_eval->_boundary_grad_variable);

    // Evaluate values
    test_fixture.evaluate<EvalType>();

    // Check values
    const auto boundary_var_result = test_fixture.getTestFieldData<EvalType>(
        fixed_eval->_boundary_variable);
    const auto boundary_grad_var_result
        = test_fixture.getTestFieldData<EvalType>(
            fixed_eval->_boundary_grad_variable);

    const int num_point = boundary_var_result.extent(1);
    const double exp_value = sign > 0 ? 2.0 : 2.75;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_value, fieldValue(boundary_var_result, 0, qp));

        for (int d = 0; d < num_space_dim; ++d)
        {
            EXPECT_DOUBLE_EQ(3.0,
                             fieldValue(boundary_grad_var_result, 0, qp, d));
        }
    }
}
//---------------------------------------------------------------------------//
// Value parameterized test fixture
struct EvaluationTest : public testing::TestWithParam<InletOutlet>
{
    // Case generator for parameterized test
    struct ParamNameGenerator
    {
        std::string
        operator()(const testing::TestParamInfo<InletOutlet>& info) const
        {
            const auto inlet_outlet = info.param;
            switch (inlet_outlet)
            {
                case (InletOutlet::inlet):
                    return "inlet";
                case (InletOutlet::outlet):
                    return "outlet";
                default:
                    return "INVALID_NAME";
            }
        }
    };
};

//---------------------------------------------------------------------------//
// Residual evaluation, 2D
TEST_P(EvaluationTest, residual2D)
{
    InletOutlet inlet_outlet;
    inlet_outlet = GetParam();
    testEval<panzer::Traits::Residual, 2>(inlet_outlet);
}

//---------------------------------------------------------------------------//
// Residual evaluation, 3D
TEST_P(EvaluationTest, residual3D)
{
    InletOutlet inlet_outlet;
    inlet_outlet = GetParam();
    testEval<panzer::Traits::Residual, 3>(inlet_outlet);
}

//---------------------------------------------------------------------------//
// Jacobian evaluation, 2D
TEST_P(EvaluationTest, jacobian2D)
{
    InletOutlet inlet_outlet;
    inlet_outlet = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(inlet_outlet);
}

//---------------------------------------------------------------------------//
// Jacobian evaluation, 3D
TEST_P(EvaluationTest, jacobian3D)
{
    InletOutlet inlet_outlet;
    inlet_outlet = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(inlet_outlet);
}

//---------------------------------------------------------------------------//
// Generate test suite with inlet and outlet conditions
INSTANTIATE_TEST_SUITE_P(InletOutletBC,
                         EvaluationTest,
                         testing::Values(InletOutlet::inlet,
                                         InletOutlet::outlet),
                         EvaluationTest::ParamNameGenerator{});

//---------------------------------------------------------------------------//
} // end namespace Test
} // namespace VertexCFD

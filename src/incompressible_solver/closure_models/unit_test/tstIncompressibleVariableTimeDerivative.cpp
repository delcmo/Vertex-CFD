#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include <incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleVariableTimeDerivative.hpp>

#include <gtest/gtest.h>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dxdt_sav;

    Dependencies(const panzer::IntegrationRule& ir)
        : _dxdt_sav("DXDT_spalart_allmaras_variable", ir.dl_scalar)
    {
        this->addEvaluatedField(_dxdt_sav);
        this->setName(
            "IncompressibleVariableTimeDerivative Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        _dxdt_sav.deep_copy(2.0);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval()
{
    // Setup test fixture.
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Eval dependencies.
    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Closure parameters
    Teuchos::ParameterList closure_params;
    closure_params.set("Field Name", "spalart_allmaras_variable");
    closure_params.set("Equation Name", "spalart_allmaras_equation");

    // Create test evaluator.
    auto dqdt_eval = Teuchos::rcp(
        new ClosureModel::IncompressibleVariableTimeDerivative<EvalType,
                                                               panzer::Traits>(
            *test_fixture.ir, closure_params));
    test_fixture.registerEvaluator<EvalType>(dqdt_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(dqdt_eval->_dqdt_var_eq);

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Check the test fields.
    const auto sae_result
        = test_fixture.getTestFieldData<EvalType>(dqdt_eval->_dqdt_var_eq);
    EXPECT_DOUBLE_EQ(2.0, fieldValue(sae_result, 0, 0));
}

//---------------------------------------------------------------------------//
TEST(IncompressibleVariableTimeDerivative, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleVariableTimeDerivative, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

} // namespace Test
} // end namespace VertexCFD

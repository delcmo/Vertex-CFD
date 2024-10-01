#include "VertexCFD_ClosureModelFactoryTestHarness.hpp"
#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_ClampField.hpp"

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test field.
template<class EvalType>
struct TestField : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                   public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    double _f;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _field;

    TestField(const panzer::IntegrationRule& ir, const double f)
        : _f(f)
        , _field("test_field", ir.dl_scalar)
    {
        this->addEvaluatedField(_field);
        this->setName("Clamp Field unit test dependency");
    }

    void evaluateFields(typename panzer::Traits::EvalData /*d*/) override
    {
        _field.deep_copy(_f);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const double f,
              const double min,
              const double max,
              const double tol,
              const double exp)
{
    // Setup test fixture.
    static constexpr int num_space_dim = 2;
    const int integration_order = 0;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create test field evaluator.
    auto dep_field_eval
        = Teuchos::rcp(new TestField<EvalType>(*test_fixture.ir, f));
    test_fixture.registerEvaluator<EvalType>(dep_field_eval);

    Teuchos::ParameterList clamp_params{"test_field"};
    clamp_params.set("Lower Limit", min)
        .set("Upper Limit", max)
        .set("Tolerance", tol);
    // Create clamped field evaluator.
    auto field_eval
        = Teuchos::rcp(new ClosureModel::ClampField<EvalType, panzer::Traits>(
            *test_fixture.ir, clamp_params));
    test_fixture.registerEvaluator<EvalType>(field_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(field_eval->_field);

    // Evaluate.
    test_fixture.evaluate<EvalType>();

    // Check the values.
    auto result = test_fixture.getTestFieldData<EvalType>(field_eval->_field);

    EXPECT_DOUBLE_EQ(exp, fieldValue(result, 0, 0));
}

//---------------------------------------------------------------------------//
// Minimum clamp test
template<class EvalType>
void testMinEval()
{
    const double field_value = -1.0;
    testEval<EvalType>(field_value, 0.1, 1.1, 1.0e-16, 0.1);
}

//---------------------------------------------------------------------------//
// Maximum clamp test
template<class EvalType>
void testMaxEval()
{
    const double field_value = 3.5;
    testEval<EvalType>(field_value, 0.1, 1.1, 1.0e-16, 1.1);
}

//---------------------------------------------------------------------------//
// Mid range clamp test
template<class EvalType>
void testMidEval()
{
    const double field_value = 0.35;
    testEval<EvalType>(field_value, 0.1, 1.1, 1.0e-16, 0.35);
}

//---------------------------------------------------------------------------//
TEST(ClampFieldMin, residual_test)
{
    testMinEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(ClampFieldMin, jacobian_test)
{
    testMinEval<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
TEST(ClampFieldMax, residual_test)
{
    testMaxEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(ClampFieldMax, jacobian_test)
{
    testMaxEval<panzer::Traits::Jacobian>();
}
//---------------------------------------------------------------------------//
TEST(ClampFieldMid, residual_test)
{
    testMidEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(ClampFieldMid, jacobian_test)
{
    testMidEval<panzer::Traits::Jacobian>();
}

} // namespace Test
} // namespace VertexCFD

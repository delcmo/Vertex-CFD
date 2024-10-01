#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType>
void testEval()
{
    static constexpr int num_space_dim = 3;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    const double value = 1.5;

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::ConstantScalarField<EvalType, panzer::Traits>(
            ir, "scalar_field", value));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_scalar_field);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_scalar_field
        = test_fixture.getTestFieldData<EvalType>(eval->_scalar_field);

    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(value, fieldValue(fv_scalar_field, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(ConstantScalarField, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(ConstantScalarField, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD

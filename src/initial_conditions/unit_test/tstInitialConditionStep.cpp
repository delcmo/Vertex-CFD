#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "initial_conditions/VertexCFD_InitialCondition_Step.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    const int integration_order = 1;
    const int basis_order = 1;
    constexpr int num_space_dim = NumSpaceDim;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    Teuchos::ParameterList params;
    const double left_value = 1.3;
    const double right_value = 2.5;
    const double origin = 0.5;
    params.set<double>("Left Value", left_value);
    params.set<double>("Right Value", right_value);
    params.set<double>("Origin", origin);
    params.set<std::string>("Equation Set Name", "dof");
    auto eval
        = Teuchos::rcp(new InitialCondition::Step<EvalType, panzer::Traits>(
            params, *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_ic);

    test_fixture.evaluate<EvalType>();

    auto ic = test_fixture.getTestFieldData<EvalType>(eval->_ic);

    // Number of degree of freedom based on `num_space_dim` value
    int num_dofs = 4;
    if (num_space_dim == 3)
        num_dofs = 8;
    EXPECT_EQ(num_dofs, ic.extent(1));

    if (num_space_dim == 2)
    {
        EXPECT_EQ(left_value, fieldValue(ic, 0, 0));
        EXPECT_EQ(right_value, fieldValue(ic, 0, 1));
        EXPECT_EQ(right_value, fieldValue(ic, 0, 2));
        EXPECT_EQ(left_value, fieldValue(ic, 0, 3));
    }
    else if (num_space_dim == 3)
    {
        EXPECT_EQ(left_value, fieldValue(ic, 0, 0));
        EXPECT_EQ(right_value, fieldValue(ic, 0, 1));
        EXPECT_EQ(right_value, fieldValue(ic, 0, 2));
        EXPECT_EQ(left_value, fieldValue(ic, 0, 3));
        EXPECT_EQ(left_value, fieldValue(ic, 0, 4));
        EXPECT_EQ(right_value, fieldValue(ic, 0, 5));
        EXPECT_EQ(right_value, fieldValue(ic, 0, 6));
        EXPECT_EQ(left_value, fieldValue(ic, 0, 7));
    }
}

//---------------------------------------------------------------------------//
TEST(Step2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(Step2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(Step3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(Step3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD

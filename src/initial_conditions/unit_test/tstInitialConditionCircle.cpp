#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "initial_conditions/VertexCFD_InitialCondition_Circle.hpp"

#include <Teuchos_Array.hpp>
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
    const double inside_value = 1.3;
    const double outside_value = 2.5;
    Teuchos::Array<double> center(num_space_dim);
    center[0] = 0.9;
    center[1] = 0.92;
    if (num_space_dim > 2)
        center[2] = 0.95;

    const double radius = 0.2;
    params.set<double>("Inside Value", inside_value);
    params.set<double>("Outside Value", outside_value);
    params.set<Teuchos::Array<double>>("Center", center);
    params.set<double>("Radius", radius);
    params.set<std::string>("Equation Set Name", "dof");
    auto eval = Teuchos::rcp(
        new InitialCondition::Circle<EvalType, panzer::Traits, num_space_dim>(
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

    // Check on values
    if (num_space_dim == 2)
    {
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 0));
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 1));
        EXPECT_EQ(inside_value, fieldValue(ic, 0, 2));
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 3));
    }
    else if (num_space_dim == 3)
    {
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 0));
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 1));
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 2));
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 3));
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 4));
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 5));
        EXPECT_EQ(inside_value, fieldValue(ic, 0, 6));
        EXPECT_EQ(outside_value, fieldValue(ic, 0, 7));
    }
}

//---------------------------------------------------------------------------//
TEST(Circle2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(Circle2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(Circle3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(Circle3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD

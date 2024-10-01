#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ExternalMagneticField.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType>
void testEval(const int num_space_dim)
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize dependents

    // Initialize class object to test
    Teuchos::ParameterList user_params;
    Teuchos::Array<double> ext_magn_vct(3);
    ext_magn_vct[0] = 1.3;
    ext_magn_vct[1] = 3.5;
    ext_magn_vct[2] = 2.4;
    user_params.set("External Magnetic Field", ext_magn_vct);
    const auto eval = Teuchos::rcp(
        new ClosureModel::ExternalMagneticField<EvalType, panzer::Traits>(
            ir, user_params));

    // Register
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < 3; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_ext_magn_field[dim]);

    test_fixture.evaluate<EvalType>();

    const auto ext_magn_field_0
        = test_fixture.getTestFieldData<EvalType>(eval->_ext_magn_field[0]);
    const auto ext_magn_field_1
        = test_fixture.getTestFieldData<EvalType>(eval->_ext_magn_field[1]);
    const auto ext_magn_field_2
        = test_fixture.getTestFieldData<EvalType>(eval->_ext_magn_field[2]);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(ext_magn_vct[0], fieldValue(ext_magn_field_0, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(ExternalMagneticField2D, residual_test)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(ExternalMagneticField2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(ExternalMagneticField3D, residual_test)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(ExternalMagneticField3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    Teuchos::Array<double> ext_magn_vct(3);
    Teuchos::ParameterList user_params;
    test_fixture.user_params.set("External Magnetic Field", ext_magn_vct);
    test_fixture.user_params.set("Build Temperature Equation", false);
    test_fixture.user_params.set("Build Electric Potential Equation", false);
    test_fixture.user_params.sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "ExternalMagneticField";
    test_fixture.eval_name = "Electric Potential External Magnetic Field";
    test_fixture.template buildAndTest<
        ClosureModel::ExternalMagneticField<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(ExternalMagneticField_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(ExternalMagneticField_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD

#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConstantSource.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType, int NumSpaceDim>
void testEval(const bool build_temp_equ)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize class object to test
    Teuchos::Array<double> mom_input_source(num_space_dim);
    mom_input_source[0] = 0.1;
    mom_input_source[1] = 0.2;
    if (num_space_dim == 3)
        mom_input_source[2] = 0.3;
    Teuchos::ParameterList user_params;
    user_params.set("Momentum Source", mom_input_source);
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 0.375);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", build_temp_equ);
    const double energy_input_source
        = build_temp_equ ? 0.4 : std::numeric_limits<double>::quiet_NaN();
    if (build_temp_equ)
    {
        user_params.set("Energy Source", energy_input_source);
        fluid_prop_list.set("Thermal conductivity", 0.5);
        fluid_prop_list.set("Specific heat capacity", 5.0);
    }
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleConstantSource<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(
            ir, fluid_prop, user_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_continuity_source);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_momentum_source[dim]);

    test_fixture.evaluate<EvalType>();

    const auto fc_cont
        = test_fixture.getTestFieldData<EvalType>(eval->_continuity_source);
    const auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_source[0]);
    const auto fc_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_source[1]);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(0.0, fieldValue(fc_cont, 0, qp));
        EXPECT_EQ(mom_input_source[0], fieldValue(fc_mom_0, 0, qp));
        EXPECT_EQ(mom_input_source[1], fieldValue(fc_mom_1, 0, qp));
        if (num_space_dim > 2) // 3D mesh
        {
            const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_momentum_source[2]);
            EXPECT_EQ(mom_input_source[2], fieldValue(fc_mom_2, 0, qp));
        }
        if (build_temp_equ)
        {
            const auto fc_energy = test_fixture.getTestFieldData<EvalType>(
                eval->_energy_source);
            EXPECT_EQ(energy_input_source, fieldValue(fc_energy, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceIsothermal2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceIsothermal2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceIsothermal3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceIsothermal3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSource3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSource3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(true);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    const Teuchos::Array<double> mom_input_source(num_space_dim);
    test_fixture.user_params.set("Momentum Source", mom_input_source);
    test_fixture.user_params.set("Build Temperature Equation", false);
    test_fixture.user_params.sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleConstantSource";
    test_fixture.eval_name = "Incompressible Constant Source "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleConstantSource<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleConstantSource_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleConstantSource_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleConstantSource_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleConstantSource_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD

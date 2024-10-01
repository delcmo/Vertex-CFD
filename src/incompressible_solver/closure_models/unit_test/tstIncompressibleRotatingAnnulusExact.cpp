#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleRotatingAnnulusExact.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Set up test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 0;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Set non-trivial quadrature point coordinates
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    ip_coord_mirror(0, 0, 0) = 2.25;
    ip_coord_mirror(0, 0, 1) = 2.20;
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Get test fixture integrator rule
    auto& ir = *test_fixture.ir;

    // Set list of parameters to pass to the test evaluator
    Teuchos::ParameterList user_params;
    user_params.set("Outer radius", 4.0);
    user_params.set("Inner radius", 2.0);
    user_params.set("Angular velocity", 0.5);
    user_params.set("Outer wall temperature", 274.0);
    user_params.set("Inner wall temperature", 273.0);

    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 50.0);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", true);
    fluid_prop_list.set("Thermal conductivity", 0.5);
    fluid_prop_list.set("Specific heat capacity", 5.0);

    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    // Create test evaluator
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleRotatingAnnulusExact<EvalType,
                                                             panzer::Traits,
                                                             num_space_dim>(
            ir, fluid_prop, user_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_velocity[dim]);
    test_fixture.registerTestField<EvalType>(eval->_temperature);

    test_fixture.evaluate<EvalType>();

    const auto fc_pressure
        = test_fixture.getTestFieldData<EvalType>(eval->_lagrange_pressure);
    const auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[0]);
    const auto fc_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[1]);
    const auto fc_energy
        = test_fixture.getTestFieldData<EvalType>(eval->_temperature);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(0.0, fieldValue(fc_pressure, 0, qp));
        EXPECT_DOUBLE_EQ(-0.8742236808886645, fieldValue(fc_mom_0, 0, qp));
        EXPECT_DOUBLE_EQ(0.8940924009088613, fieldValue(fc_mom_1, 0, qp));
        if (num_space_dim == 3)
        {
            const auto fc_mom_2
                = test_fixture.getTestFieldData<EvalType>(eval->_velocity[2]);
            EXPECT_EQ(0.0, fieldValue(fc_mom_2, 0, qp));
        }

        EXPECT_EQ(292.43421676378887, fieldValue(fc_energy, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleRotatingAnnulusExact2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleRotatingAnnulusExact2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleRotatingAnnulusExact3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleRotatingAnnulusExact3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "IncompressibleRotatingAnnulusExact";
    test_fixture.eval_name = "Exact Solution Rotating Annulus";
    test_fixture.user_params.sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.user_params.set("Outer radius", 2.0)
        .set("Inner radius", 3.0)
        .set("Angular velocity", 4.0)
        .set("Outer wall temperature", 5.0)
        .set("Inner wall temperature", 6.0);
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleRotatingAnnulusExact<EvalType,
                                                         panzer::Traits,
                                                         num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleRotatingAnnulusExact_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleRotatingAnnulusExact_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleRotatingAnnulusExact_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleRotatingAnnulusExact_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD

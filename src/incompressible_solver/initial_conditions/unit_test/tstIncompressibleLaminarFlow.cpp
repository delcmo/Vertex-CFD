#include <VertexCFD_EvaluatorTestHarness.hpp>

#include "incompressible_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleLaminarFlow.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool build_temp_equ)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Set non-trivial coordinates for the degree of freedom
    const int num_coord = test_fixture.cell_topo->getNodeCount();
    Kokkos::View<double**, Kokkos::HostSpace> x(
        "coordinate", num_space_dim, num_coord);
    auto basis_coord_view
        = test_fixture.workset->bases[0]->basis_coordinates.get_static_view();
    auto basis_coord_mirror = Kokkos::create_mirror(basis_coord_view);
    for (int basis = 0; basis < num_coord; basis++)
    {
        // random coordinate assigned
        basis_coord_mirror(0, basis, 1) = 0.125 * (basis + 1);
        if (num_space_dim == 3)
            basis_coord_mirror(0, basis, 2) = -2.0 + 0.125 * (basis + 1);
    }

    Kokkos::deep_copy(basis_coord_view, basis_coord_mirror);

    Teuchos::ParameterList ic_params;
    ic_params.set("Minimum height", -2.0);
    ic_params.set("Maximum height", 2.2);
    ic_params.set("Average velocity", 3.0);
    const double temp_ref
        = build_temp_equ ? 4.0 : std::numeric_limits<double>::quiet_NaN();
    if (build_temp_equ)
        ic_params.set("Temperature", temp_ref);
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 0.375);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", build_temp_equ);
    if (build_temp_equ)
    {
        fluid_prop_list.set("Thermal conductivity", 0.5);
        fluid_prop_list.set("Specific heat capacity", 0.6);
    }
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    // Register and evaluate fields
    auto eval = Teuchos::rcp(
        new InitialCondition::IncompressibleLaminarFlow<EvalType,
                                                        panzer::Traits,
                                                        num_space_dim>(
            ic_params, fluid_prop, *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);

    test_fixture.registerTestField<EvalType>(eval->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_velocity[dim]);
    if (build_temp_equ)
        test_fixture.registerTestField<EvalType>(eval->_temperature);

    test_fixture.evaluate<EvalType>();

    const auto phi
        = test_fixture.getTestFieldData<EvalType>(eval->_lagrange_pressure);
    const auto u = test_fixture.getTestFieldData<EvalType>(eval->_velocity[0]);
    const auto v = test_fixture.getTestFieldData<EvalType>(eval->_velocity[1]);

    // Check number of degree of freedoms
    const int num_dofs = num_space_dim == 2 ? 4 : 8;
    EXPECT_EQ(num_dofs, phi.extent(1));
    EXPECT_EQ(num_dofs, u.extent(1));
    EXPECT_EQ(num_dofs, v.extent(1));

    // Reference values
    const double phi_ref = 0.0;
    const double u_ref_2d[4] = {4.48405612244898,
                                4.436224489795919,
                                4.356505102040816,
                                4.244897959183674};
    const double u_ref_3d[8] = {1.195578231292517,
                                1.7482993197278913,
                                2.2159863945578238,
                                2.5986394557823127,
                                2.896258503401361,
                                3.1088435374149666,
                                3.2363945578231297,
                                3.278911564625851};
    const double v_ref = 0.0;
    const double w_ref = 0.0;

    // Check values
    for (int n = 0; n < num_dofs; ++n)
    {
        EXPECT_DOUBLE_EQ(phi_ref, fieldValue(phi, 0, n));
        const double u_ref = num_space_dim == 2 ? u_ref_2d[n] : u_ref_3d[n];
        EXPECT_FLOAT_EQ(u_ref, fieldValue(u, 0, n));
        EXPECT_FLOAT_EQ(v_ref, fieldValue(v, 0, n));
        if (num_space_dim == 3)
        {
            const auto w
                = test_fixture.getTestFieldData<EvalType>(eval->_velocity[2]);
            EXPECT_EQ(num_dofs, w.extent(1));
            EXPECT_FLOAT_EQ(w_ref, fieldValue(w, 0, n));
        }
        if (build_temp_equ)
        {
            const auto temp
                = test_fixture.getTestFieldData<EvalType>(eval->_temperature);
            EXPECT_EQ(num_dofs, temp.extent(1));
            EXPECT_FLOAT_EQ(temp_ref, fieldValue(temp, 0, n));
        }
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLaminarFLowIsothermal2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLaminarFLowIsothermal2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLaminarFLowIsothermal3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLaminarFLowIsothermal3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLaminarFLow3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(true);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLaminarFLow3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(true);
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD

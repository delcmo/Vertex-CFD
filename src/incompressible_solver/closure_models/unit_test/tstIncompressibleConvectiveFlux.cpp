#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConvectiveFlux.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    double _u;
    double _v;
    double _w;
    bool _build_temp;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> temperature;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w,
                 const bool build_temp)
        : _u(u)
        , _v(v)
        , _w(w)
        , _build_temp(build_temp)
        , lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)
        , temperature("temperature", ir.dl_scalar)

    {
        this->addEvaluatedField(lagrange_pressure);
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);
        if (_build_temp)
            this->addEvaluatedField(temperature);

        this->setName("Incompressible Convective Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        lagrange_pressure.deep_copy(0.75);
        vel_0.deep_copy(_u);
        vel_1.deep_copy(_v);
        vel_2.deep_copy(_w);
        if (_build_temp)
            temperature.deep_copy(_u + _v);
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool unscaled_density, const bool build_temp)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const double u = 0.25;
    const double v = 0.5;
    const double w
        = num_space_dim > 2 ? 0.125 : std::numeric_limits<double>::quiet_NaN();

    auto deps
        = Teuchos::rcp(new Dependencies<EvalType>(ir, u, v, w, build_temp));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    double rho = 1.0;
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 0.375);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", build_temp);
    if (unscaled_density)
    {
        rho = 3.0;
        fluid_prop_list.set("Density", rho);
    }
    if (build_temp)
    {
        fluid_prop_list.set("Thermal conductivity", 0.5);
        fluid_prop_list.set("Specific heat capacity", 5.0);
    }
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleConvectiveFlux<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(
            ir, fluid_prop));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_continuity_flux);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_momentum_flux[dim]);

    test_fixture.evaluate<EvalType>();

    auto fc_cont
        = test_fixture.getTestFieldData<EvalType>(eval->_continuity_flux);
    auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[0]);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_cont_flux[3] = {rho * u, rho * v, rho * w};
    const double exp_mom_0_flux[3]
        = {unscaled_density ? 0.9375 : 0.8125,
           unscaled_density ? 0.375 : 0.125,
           num_space_dim == 3 ? unscaled_density ? 0.09375 : 0.03125
                              : std::numeric_limits<double>::quiet_NaN()};
    const double exp_mom_1_flux[3]
        = {unscaled_density ? 0.375 : 0.125,
           unscaled_density ? 1.5 : 1.,
           num_space_dim == 3 ? unscaled_density ? 0.1875 : 0.0625
                              : std::numeric_limits<double>::quiet_NaN()};
    const double exp_mom_2_flux[3]
        = {unscaled_density ? 0.09375 : 0.03125,
           unscaled_density ? 0.1875 : 0.0625,
           num_space_dim == 3 ? unscaled_density ? 0.796875 : 0.765625
                              : std::numeric_limits<double>::quiet_NaN()};
    const double exp_ener_flux[3]
        = {unscaled_density ? 2.8125 : 0.9375,
           unscaled_density ? 5.625 : 1.875,
           num_space_dim == 3 ? unscaled_density ? 1.40625 : 0.46875
                              : std::numeric_limits<double>::quiet_NaN()};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_EQ(exp_cont_flux[dim], fieldValue(fc_cont, 0, qp, dim));
            EXPECT_EQ(exp_mom_0_flux[dim], fieldValue(fc_mom_0, 0, qp, dim));
            const auto fc_mom_1 = test_fixture.getTestFieldData<EvalType>(
                eval->_momentum_flux[1]);
            EXPECT_EQ(exp_mom_1_flux[dim], fieldValue(fc_mom_1, 0, qp, dim));
            if (num_space_dim > 2) // 3D mesh
            {
                const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                    eval->_momentum_flux[2]);
                EXPECT_EQ(exp_mom_2_flux[dim],
                          fieldValue(fc_mom_2, 0, qp, dim));
            }
            if (build_temp)
            {
                const auto fc_energy = test_fixture.getTestFieldData<EvalType>(
                    eval->_energy_flux);
                EXPECT_EQ(exp_ener_flux[dim],
                          fieldValue(fc_energy, 0, qp, dim));
            }
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermal2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermal2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensity2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(false, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensity2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(false, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermal3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermal3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityIsothermal3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(true, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityIsothermal3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(true, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensity3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(true, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensity3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.user_params.set("Build Temperature Equation", false);
    test_fixture.user_params.sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleConvectiveFlux";
    test_fixture.eval_name = "Incompressible Convective Flux "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleConvectiveFlux<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleConvectiveFlux_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleConvectiveFlux_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleConvectiveFlux_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleConvectiveFlux_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD

#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricPotentialDiffusionFlux.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_electric_potential;

    Dependencies(const panzer::IntegrationRule& ir, std::string grad_prefix)
        : grad_electric_potential(grad_prefix + "GRAD_electric_potential",
                                  ir.dl_vector)
    {
        this->addEvaluatedField(grad_electric_potential);
        this->setName(
            "Electric Potential Diffusion Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "electric potential diffusion flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_electric_potential.extent(1);
        const int num_grad_dim = grad_electric_potential.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
            for (int dim = 0; dim < num_grad_dim; ++dim)
                grad_electric_potential(c, qp, dim) = 1.5 * (qp + 1)
                                                      * (dim + 1);
    }
};

//-----------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_grad_dim,
              const std::string flux_prefix = "",
              const std::string grad_prefix = "")
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, grad_prefix));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize closure model constructor
    const double sigma = 3.0;
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 0.375);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", false);
    fluid_prop_list.set("Build Electric Potential Equation", true);
    fluid_prop_list.set("Electrical conductivity", sigma);
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    const auto eval = Teuchos::rcp(
        new ClosureModel::ElectricPotentialDiffusionFlux<EvalType, panzer::Traits>(
            ir, fluid_prop, flux_prefix, grad_prefix));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_electric_potential_flux);
    test_fixture.evaluate<EvalType>();

    const auto diff_flux = test_fixture.getTestFieldData<EvalType>(
        eval->_electric_potential_flux);

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_grad_dim; dim++)
        {
            const auto exp_value = sigma * 1.5 * (qp + 1) * (dim + 1);
            EXPECT_EQ(-exp_value, fieldValue(diff_flux, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(ElectricPotentialDiffusion2D, residual_test)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(ElectricPotentialDiffusion2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(ElectricPotentialDiffusion3D, residual_test)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(ElectricPotentialDiffusion3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//-----------------------------------------------------------------//
TEST(ElectricPotentialDiffusion3DPrefix, residual_test)
{
    testEval<panzer::Traits::Residual>(3, "BAR_", "BOO_");
}

//-----------------------------------------------------------------//
TEST(ElectricPotentialDiffusion3DPrefix, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(3, "BAR_", "BOO_");
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
        .set("Artificial compressibility", 2.0)
        .set("Build Electric Potential Equation", false);
    test_fixture.type_name = "ElectricPotentialDiffusionFlux";
    test_fixture.eval_name = "Electric Potential Diffusion Flux "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::ElectricPotentialDiffusionFlux<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(ElectricPotentialFlux_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(ElectricPotentialFlux_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD

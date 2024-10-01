#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaDiffusivityCoefficient.hpp"

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

    double _k;
    double _w;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _turb_kinetic_energy;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double k,
                 const double w)
        : _k(k)
        , _w(w)
        , _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
        , _turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                          ir.dl_scalar)
    {
        this->addEvaluatedField(_turb_kinetic_energy);
        this->addEvaluatedField(_turb_specific_dissipation_rate);
        this->setName(
            "K-Omega Incompressible Diffusivity Coefficient Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        _turb_kinetic_energy.deep_copy(_k);
        _turb_specific_dissipation_rate.deep_copy(_w);
    }
};

template<class EvalType>
void testEval()
{
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Set turbulent quantities
    const double k_value = 1.5;
    const double w_value = 3.5;
    const double sigma_k = 0.7;
    const double sigma_w = 0.65;

    // Create parameter list for user-defined constants
    Teuchos::ParameterList user_params;
    user_params.sublist("Turbulence Parameters")
        .sublist("K-Omega Parameters")
        .set<double>("sigma_k", sigma_k);
    user_params.sublist("Turbulence Parameters")
        .sublist("K-Omega Parameters")
        .set<double>("sigma_w", sigma_w);

    // Eval dependencies
    const auto deps
        = Teuchos::rcp(new Dependencies<EvalType>(ir, k_value, w_value));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Fluid properties
    const double nu = 0.25;
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", nu);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", false);
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleKOmegaDiffusivityCoefficient<
            EvalType,
            panzer::Traits>(ir, fluid_prop, user_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_diffusivity_var_k);
    test_fixture.registerTestField<EvalType>(eval->_diffusivity_var_w);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_var_k
        = test_fixture.getTestFieldData<EvalType>(eval->_diffusivity_var_k);
    const auto fv_var_w
        = test_fixture.getTestFieldData<EvalType>(eval->_diffusivity_var_w);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_diffusivity_var_k = nu + sigma_k * k_value / w_value;
    const double exp_diffusivity_var_w = nu + sigma_w * k_value / w_value;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_diffusivity_var_k, fieldValue(fv_var_k, 0, qp));
        EXPECT_DOUBLE_EQ(exp_diffusivity_var_w, fieldValue(fv_var_w, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaDiffusivityCoefficient, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaDiffusivityCoefficient, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD

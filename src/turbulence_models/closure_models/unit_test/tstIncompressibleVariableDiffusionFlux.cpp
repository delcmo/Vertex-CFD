#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include <turbulence_models/closure_models/VertexCFD_Closure_IncompressibleVariableDiffusionFlux.hpp>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_var;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> diffusivity_var;

    Dependencies(const panzer::IntegrationRule& ir)
        : grad_var("GRAD_spalart_allmaras_variable", ir.dl_vector)
        , diffusivity_var("diffusivity_spalart_allmaras_variable", ir.dl_scalar)
    {
        this->addEvaluatedField(grad_var);
        this->addEvaluatedField(diffusivity_var);
        this->setName(
            "Incompressible Variable Diffusion Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        diffusivity_var.deep_copy(2.0);

        Kokkos::parallel_for(
            "incompressible variable diffusion flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_var.extent(1);
        const int num_space_dim = grad_var.extent(2);
        using std::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                grad_var(c, qp, dim) = 13.0 * (dim + 1);
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Closure parameters
    Teuchos::ParameterList closure_params;
    closure_params.set("Field Name", "spalart_allmaras_variable");
    closure_params.set("Equation Name", "spalart_allmaras_equation");

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleVariableDiffusionFlux<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>(
            ir, closure_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_var_diff_flux);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_var
        = test_fixture.getTestFieldData<EvalType>(eval->_var_diff_flux);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_var_diff_flux[3] = {
        26.0,
        52.0,
        num_space_dim == 2 ? std::numeric_limits<double>::quiet_NaN() : 78.0};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            EXPECT_EQ(exp_var_diff_flux[dim], fieldValue(fv_var, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleVariableDiffusionFlux2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleVariableDiffusionFlux2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleVariableDiffusionFlux3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleVariableDiffusionFlux3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD

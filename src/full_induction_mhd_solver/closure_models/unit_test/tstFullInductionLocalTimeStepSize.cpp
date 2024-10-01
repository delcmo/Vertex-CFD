#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_FullInductionLocalTimeStepSize.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    double _u;
    double _v;
    double _w;
    Kokkos::Array<double, 3> _h;

    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> _element_length;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>, 3>
        _velocity;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>, 3>
        _tot_magn_field;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w,
                 const Kokkos::Array<double, 3> h)
        : _u(u)
        , _v(v)
        , _w(w)
        , _h(h)
        , _element_length("element_length", ir.dl_vector)
    {
        this->addEvaluatedField(_element_length);
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, _velocity, "velocity_");
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, _tot_magn_field, "total_magnetic_field_");
        this->setName(
            "Full Induction Local Time Step Size Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _velocity[0].deep_copy(_u);
        _velocity[1].deep_copy(_v);
        _velocity[2].deep_copy(_w);
        _tot_magn_field[0].deep_copy(1.1);
        _tot_magn_field[1].deep_copy(-1.2);
        _tot_magn_field[2].deep_copy(1.3);
        Kokkos::parallel_for(
            "test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        int num_point = _element_length.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            _element_length(c, qp, 0) = _h[0];
            _element_length(c, qp, 1) = _h[1];
            _element_length(c, qp, 2) = _h[2];
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool build_magn_corr)
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Eval dependencies.
    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    const double u = -1.5;
    const double v = 2.0;
    const double w = num_space_dim == 3 ? -0.5 : nan_val;
    const Kokkos::Array<double, 3> h
        = {0.25, 0.5, num_space_dim == 3 ? 0.75 : nan_val};

    auto dep_eval = Teuchos::rcp(
        new Dependencies<EvalType>(*test_fixture.ir, u, v, w, h));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    Teuchos::ParameterList fluid_params;
    fluid_params.set("Density", 0.9);
    fluid_params.set("Kinematic viscosity", nan_val);
    fluid_params.set("Artificial compressibility", nan_val);
    fluid_params.set("Build Temperature Equation", false);
    FluidProperties::ConstantFluidProperties incompressible_fluidprop_params
        = FluidProperties::ConstantFluidProperties(fluid_params);

    const double c_h = num_space_dim == 2 ? 0.1 : 5.0;
    Teuchos::ParameterList full_induction_params;
    full_induction_params.set("Vacuum Magnetic Permeability", 0.1);
    full_induction_params.set("Build Magnetic Correction Potential Equation",
                              build_magn_corr);
    full_induction_params.set("Hyperbolic Divergence Cleaning Speed", c_h);
    MHDProperties::FullInductionMHDProperties mhd_props
        = MHDProperties::FullInductionMHDProperties(full_induction_params);

    // Create test evaluator.
    auto dt_eval = Teuchos::rcp(
        new ClosureModel::FullInductionLocalTimeStepSize<EvalType,
                                                         panzer::Traits,
                                                         NumSpaceDim>(
            *test_fixture.ir, incompressible_fluidprop_params, mhd_props));
    test_fixture.registerEvaluator<EvalType>(dt_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(dt_eval->_local_dt);

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Check the test fields.
    auto local_dt_result
        = test_fixture.getTestFieldData<EvalType>(dt_eval->_local_dt);

    const int num_qp = num_space_dim == 2 ? 4 : 8;
    const double result = num_space_dim == 2 ? 0.030612244897959186
                          : build_magn_corr  ? 0.02112676056338028
                                             : 0.02556818181818182;
    for (int qp = 0; qp < num_qp; ++qp)
        EXPECT_DOUBLE_EQ(result, fieldValue(local_dt_result, 0, qp));
}

//---------------------------------------------------------------------------//
TEST(FullInductionLocalTimeStepSize2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(true);
}

//---------------------------------------------------------------------------//
TEST(FullInductionLocalTimeStepSize2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(true);
}

//---------------------------------------------------------------------------//
TEST(FullInductionLocalTimeStepSize3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(true);
}

//---------------------------------------------------------------------------//
TEST(FullInductionLocalTimeStepSize3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(true);
}

//---------------------------------------------------------------------------//
TEST(FullInductionLocalTimeStepSizeNoDivCleaning3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(false);
}

//---------------------------------------------------------------------------//
TEST(FullInductionLocalTimeStepSizeNoDivCleaning3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(false);
}

//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "FullInductionLocalTimeStepSize";
    test_fixture.eval_name = "Full Induction Local Time Step Size";
    test_fixture.user_params.sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.user_params.sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.1)
        .set("Build Magnetic Correction Potential Equation", false);
    test_fixture.template buildAndTest<
        ClosureModel::FullInductionLocalTimeStepSize<EvalType,
                                                     panzer::Traits,
                                                     num_space_dim>,
        num_space_dim>();
}

TEST(FullInductionLocalTimeStepSize_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(FullInductionLocalTimeStepSize_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // end namespace Test
} // end namespace VertexCFD

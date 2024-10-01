#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MagneticPressure.hpp"

#include "utils/VertexCFD_Utils_VectorField.hpp"

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

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>, 3>
        tot_magn_field;

    const Kokkos::Array<double, 3> _b;

    Dependencies(const panzer::IntegrationRule& ir,
                 const Kokkos::Array<double, 3> b)
        : _b(b)
    {
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, tot_magn_field, "total_magnetic_field_");
        this->setName("Magnetic Pressure");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        const int num_field_dim = tot_magn_field.size();
        for (int dim = 0; dim < num_field_dim; ++dim)
            tot_magn_field[dim].deep_copy(_b[dim]);
    }
};

template<class EvalType>
void testEval(const int num_space_dim)
{
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize magnetic field components and dependencies
    const double b0 = 0.25;
    const double b1 = 0.5;
    const double b2 = num_space_dim == 3 ? 0.75 : 0.125;

    // Eval dependencies
    const auto deps
        = Teuchos::rcp(new Dependencies<EvalType>(ir, {b0, b1, b2}));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Closure parameters
    const double mu_0 = 2.0e-3;
    Teuchos::ParameterList full_induction_params;
    full_induction_params.set("Vacuum Magnetic Permeability", mu_0);
    MHDProperties::FullInductionMHDProperties mhd_props
        = MHDProperties::FullInductionMHDProperties(full_induction_params);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::MagneticPressure<EvalType, panzer::Traits>(
            ir, mhd_props));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_magnetic_pressure);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_magn_pres
        = test_fixture.getTestFieldData<EvalType>(eval->_magnetic_pressure);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_magn_pres = (b0 * b0 + b1 * b1 + b2 * b2) / (2.0 * mu_0);

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_magn_pres, fieldValue(fv_magn_pres, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(MagneticPressure2D, residual_test)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(MagneticPressure2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(MagneticPressure3D, residual_test)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(MagneticPressure3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    static constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "MagneticPressure";
    test_fixture.eval_name = "Magnetic Pressure";
    test_fixture.user_params.sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.125);
    test_fixture.user_params.sublist("Fluid Properties")
        .set("Kinematic viscosity", 1.5)
        .set("Artificial compressibility", 0.1);
    test_fixture.template buildAndTest<
        ClosureModel::MagneticPressure<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(MagneticPressure_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(MagneticPressure_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(MagneticPressure_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(MagneticPressure_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD

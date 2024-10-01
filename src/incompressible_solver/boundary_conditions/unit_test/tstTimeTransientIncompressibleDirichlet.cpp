#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleDirichlet.hpp"
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

#include <mpi.h>

#include <iostream>

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

    double _phi, _u0, _u1, _u2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_2;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double phi,
                 const double u0,
                 const double u1,
                 const double u2)
        : _phi(phi)
        , _u0(u0)
        , _u1(u1)
        , _u2(u2)
        , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , _grad_velocity_0("GRAD_velocity_0", ir.dl_vector)
        , _grad_velocity_1("GRAD_velocity_1", ir.dl_vector)
        , _grad_velocity_2("GRAD_velocity_2", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);
        this->setName(
            "Time Transient Incompressible Dirichlet Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        _lagrange_pressure.deep_copy(_phi);
        _grad_velocity_0.deep_copy(_u0);
        _grad_velocity_1.deep_copy(_u1);
        _grad_velocity_2.deep_copy(_u2);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const Kokkos::Array<double, 3> time_values,
              const Kokkos::Array<double, NumSpaceDim> final_values,
              const Kokkos::Array<double, NumSpaceDim> init_values,
              const Kokkos::Array<double, NumSpaceDim> exp_fields)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int num_grad_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    // Create dependencies
    const double phi = 3.0;
    const double u0 = final_values[0];
    const double u1 = final_values[1];
    const double u2 = num_space_dim == 3 ? final_values[2] : 0.0;
    const double u0_init = init_values[0];
    const double u1_init = init_values[1];
    const double u2_init = num_space_dim == 3 ? init_values[2] : -1.0;
    const double time_init = time_values[0];
    const double time_final = time_values[1];
    const double time = time_values[2];

    auto dep_eval = Teuchos::rcp(
        new Dependencies<EvalType>(*test_fixture.ir, phi, u0, u1, u2));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Equation of state
    const bool build_temp_equ = false;
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

    // Create the param list to initialize the evaluator
    Teuchos::ParameterList bc_params;
    bc_params.set("velocity_0", u0);
    bc_params.set("velocity_1", u1);
    if (num_space_dim == 3)
        bc_params.set("velocity_2", u2);
    if (u0_init > 0.0)
        bc_params.set("velocity_0_init", u0_init);
    if (u1_init > 0.0)
        bc_params.set("velocity_1_init", u1_init);
    if (u2_init > 0.0)
        bc_params.set("velocity_2_init", u2_init);
    if (time_init > 0.0)
        bc_params.set("Time Initial", time_init);
    if (time_final > 0.0)
        bc_params.set("Time Final", time_final);

    // Create dirichlet evaluator.
    auto dirichlet_eval = Teuchos::rcp(
        new BoundaryCondition::
            IncompressibleDirichlet<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir, fluid_prop, bc_params));
    test_fixture.registerEvaluator<EvalType>(dirichlet_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        dirichlet_eval->_boundary_lagrange_pressure);
    for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
    {
        test_fixture.registerTestField<EvalType>(
            dirichlet_eval->_boundary_velocity[vel_dim]);
        test_fixture.registerTestField<EvalType>(
            dirichlet_eval->_boundary_grad_velocity[vel_dim]);
    }

    // Set time
    test_fixture.setTime(time);

    // Evaluate time transient dirichlet BC.
    test_fixture.evaluate<EvalType>();

    // Check the time transient dirichlet BC.
    const auto boundary_phi_result = test_fixture.getTestFieldData<EvalType>(
        dirichlet_eval->_boundary_lagrange_pressure);

    const int num_point = boundary_phi_result.extent(1);

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(phi, fieldValue(boundary_phi_result, 0, qp));
        for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    dirichlet_eval->_boundary_velocity[vel_dim]);
            EXPECT_DOUBLE_EQ(exp_fields[vel_dim],
                             fieldValue(boundary_velocity_d_result, 0, qp));
        }

        for (int d = 0; d < num_grad_dim; ++d)
        {
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                // The gradients are initialized with entries from
                // `final_values`.
                const double exp_val = final_values[vel_dim];
                const auto boundary_grad_velocity_d_result
                    = test_fixture.getTestFieldData<EvalType>(
                        dirichlet_eval->_boundary_grad_velocity[vel_dim]);
                EXPECT_DOUBLE_EQ(
                    exp_val,
                    fieldValue(boundary_grad_velocity_d_result, 0, qp, d));
            }
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D: time transient dirichlet - steady
template<class EvalType>
void testTimeTransientDirichletSteady2D()
{
    const Kokkos::Array<double, 3> time_values = {-0.5, -2.0, 3.0};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {-1.0, -1.2};
    const Kokkos::Array<double, 2> exp_fields = {2.0, 3.0};

    testEval<EvalType, 2>(time_values, final_values, init_values, exp_fields);
}

// 2-D time transient dirichlet residual
TEST(TimeTransientDirichletSteady2D, residual)
{
    testTimeTransientDirichletSteady2D<panzer::Traits::Residual>();
}

// 2-D time transient dirichlet jacobian
TEST(TimeTransientDirichletSteady2D, jacobian)
{
    testTimeTransientDirichletSteady2D<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// 2-D time transient dirichlet - time > time_final
template<class EvalType>
void testTimeTransientDirichletTimeFinal2D()
{
    const Kokkos::Array<double, 3> time_values = {0.5, 2.0, 3.0};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {1.0, 1.2};
    const Kokkos::Array<double, 2> exp_fields = {2.0, 3.0};

    testEval<EvalType, 2>(time_values, final_values, init_values, exp_fields);
}

// 2-D time transient dirichlet residual
TEST(TimeTransientDirichletTimeFinal2D, residual)
{
    testTimeTransientDirichletTimeFinal2D<panzer::Traits::Residual>();
}

// 2-D time transient dirichlet jacobian
TEST(TimeTransientDirichletTimeFinal2D, jacobian)
{
    testTimeTransientDirichletTimeFinal2D<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// 2-D time transient dirichlet - time < time_init
template<class EvalType>
void testTimeTransientDirichletTimeInit2D()
{
    const Kokkos::Array<double, 3> time_values = {0.5, 1.0, 0.2};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {1.0, 1.2};
    const Kokkos::Array<double, 2> exp_fields = {1.0, 1.2};

    testEval<EvalType, 2>(time_values, final_values, init_values, exp_fields);
}

// 2-D time transient dirichlet residual
TEST(TimeTransientDirichletTimeInit2D, residual)
{
    testTimeTransientDirichletTimeInit2D<panzer::Traits::Residual>();
}

// 2-D time transient dirichlet jacobian
TEST(TimeTransientDirichletTimeInit2D, jacobian)
{
    testTimeTransientDirichletTimeInit2D<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// 2-D time transient dirichlet - time_init < time < time_final. The expected
// values should be the average value between the final and the initial values.
template<class EvalType>
void testTimeTransientDirichletTimeIntermediate2D()
{
    const Kokkos::Array<double, 3> time_values = {1.0, 2.0, 1.5};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {1.0, 2.0};
    const Kokkos::Array<double, 2> exp_fields = {1.5, 2.5};

    testEval<EvalType, 2>(time_values, final_values, init_values, exp_fields);
}

// 2-D time transient dirichlet residual
TEST(TimeTransientDirichletTimeIntermediate2D, residual)
{
    testTimeTransientDirichletTimeIntermediate2D<panzer::Traits::Residual>();
}

// 2-D time transient dirichlet jacobian
TEST(TimeTransientDirichletTimeIntermediate2D, jacobian)
{
    testTimeTransientDirichletTimeIntermediate2D<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// 3-D time transient dirichlet - time_init < time < time_final. The expected
// values should be the average value between the final and the initial values
template<class EvalType>
void testTimeTransientDirichletTimeIntermediate3D()
{
    const Kokkos::Array<double, 3> time_values = {1.0, 2.0, 1.5};
    const Kokkos::Array<double, 3> final_values = {2.0, 3.0, 4.0};
    const Kokkos::Array<double, 3> init_values = {1.0, 2.0, 3.0};
    const Kokkos::Array<double, 3> exp_fields = {1.5, 2.5, 3.5};

    testEval<EvalType, 3>(time_values, final_values, init_values, exp_fields);
}

// 3-D time transient dirichlet residual
TEST(TimeTransientDirichletTimeIntermediate3D, residual)
{
    testTimeTransientDirichletTimeIntermediate3D<panzer::Traits::Residual>();
}

// 3-D time transient dirichlet jacobian
TEST(TimeTransientDirichletTimeIntermediate3D, jacobian)
{
    testTimeTransientDirichletTimeIntermediate2D<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD

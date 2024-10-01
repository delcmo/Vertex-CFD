#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleRotatingWall.hpp"
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_ParameterList.hpp>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_2;

    bool _build_tmp_equ;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;

    Dependencies(const panzer::IntegrationRule& ir, const bool build_tmp_equ)
        : _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , _grad_velocity_0("GRAD_velocity_0", ir.dl_vector)
        , _grad_velocity_1("GRAD_velocity_1", ir.dl_vector)
        , _grad_velocity_2("GRAD_velocity_2", ir.dl_vector)
        , _build_tmp_equ(build_tmp_equ)
        , _grad_temperature("GRAD_temperature", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);

        if (build_tmp_equ)
            this->addEvaluatedField(_grad_temperature);

        this->setName(
            "Time Transient Incompressible Rotating Wall Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        // Initialize pressure
        _lagrange_pressure.deep_copy(0.4);

        // Initialize gradients
        Kokkos::parallel_for(
            "incompressible rotating wall test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _lagrange_pressure.extent(1);
        const int num_space_dim = _grad_velocity_0.extent(2);

        // Loop over quadrature points and mesh dimension
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int d = 0; d < num_space_dim; d++)
            {
                const int dqp = (d + 1 + num_space_dim) * (qp + 1);
                _grad_velocity_0(c, qp, d) = 0.1 * dqp;
                _grad_velocity_1(c, qp, d) = 0.2 * dqp;
                if (num_space_dim == 3)
                    _grad_velocity_2(c, qp, d) = 0.3 * dqp;

                if (_build_tmp_equ)
                    _grad_temperature(c, qp, d) = 325 * dqp;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const Kokkos::Array<double, 3> time_values,
              const Kokkos::Array<double, 1> init_values,
              const Kokkos::Array<double, NumSpaceDim> exp_fields,
              const bool build_temp_equ)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int num_grad_dim = num_space_dim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    // Create dependencies
    const double angular_velocity = 2.0;
    const double angular_velocity_init = init_values[0];
    const double time_init = time_values[0];
    const double time_final = time_values[1];
    const double time = time_values[2];

    // Initialize values
    const double T_wall
        = build_temp_equ ? 325 : std::numeric_limits<double>::quiet_NaN();

    // Set non-trivial quadrature points to avoid x = y
    test_fixture.int_values->ip_coordinates(0, 0, 0) = 0.7375;
    test_fixture.int_values->ip_coordinates(0, 0, 1) = 0.9775;

    // Initialize dependecy evaluator
    auto dep_eval = Teuchos::rcp(
        new Dependencies<EvalType>(*test_fixture.ir, build_temp_equ));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Thermophysical properties
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
    bc_params.set("Angular Velocity", angular_velocity);
    if (init_values[0] > 0.0)
        bc_params.set("Angular Velocity Initial", angular_velocity_init);
    if (time_init > 0.0)
        bc_params.set("Time Initial", time_init);
    if (time_final > 0.0)
        bc_params.set("Time Final", time_final);
    if (build_temp_equ)
        bc_params.set("Wall Temperature", T_wall);

    // Create evaluator.
    auto isotherm_eval = Teuchos::rcp(
        new BoundaryCondition::
            IncompressibleRotatingWall<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir, fluid_prop, bc_params));
    test_fixture.registerEvaluator<EvalType>(isotherm_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        isotherm_eval->_boundary_lagrange_pressure);
    for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
    {
        test_fixture.registerTestField<EvalType>(
            isotherm_eval->_boundary_velocity[vel_dim]);
        test_fixture.registerTestField<EvalType>(
            isotherm_eval->_boundary_grad_velocity[vel_dim]);
    }

    if (build_temp_equ)
    {
        test_fixture.registerTestField<EvalType>(
            isotherm_eval->_boundary_temperature);
        test_fixture.registerTestField<EvalType>(
            isotherm_eval->_boundary_grad_temperature);
    }

    // Set time
    test_fixture.setTime(time);

    // Evaluate
    test_fixture.evaluate<EvalType>();

    // Get field
    const auto boundary_phi_result = test_fixture.getTestFieldData<EvalType>(
        isotherm_eval->_boundary_lagrange_pressure);

    // Assert variables and gradients at each quadrature points
    const int num_point = boundary_phi_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(0.4, fieldValue(boundary_phi_result, 0, qp));

        for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    isotherm_eval->_boundary_velocity[vel_dim]);
            EXPECT_DOUBLE_EQ(exp_fields[vel_dim],
                             fieldValue(boundary_velocity_d_result, 0, qp));
        }

        if (build_temp_equ)
        {
            const auto boundary_temperature_result
                = test_fixture.getTestFieldData<EvalType>(
                    isotherm_eval->_boundary_temperature);
            EXPECT_DOUBLE_EQ(T_wall,
                             fieldValue(boundary_temperature_result, 0, qp));
        }

        for (int d = 0; d < num_grad_dim; ++d)
        {
            const int dqp = (d + 1 + num_space_dim) * (qp + 1);
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                const auto boundary_grad_velocity_d_result
                    = test_fixture.getTestFieldData<EvalType>(
                        isotherm_eval->_boundary_grad_velocity[vel_dim]);
                EXPECT_DOUBLE_EQ(
                    (vel_dim + 1) * 0.1 * dqp,
                    fieldValue(boundary_grad_velocity_d_result, 0, qp, d));

                if (build_temp_equ)
                {
                    const auto boundary_grad_temperature_result
                        = test_fixture.getTestFieldData<EvalType>(
                            isotherm_eval->_boundary_grad_temperature);
                    EXPECT_DOUBLE_EQ(
                        T_wall * dqp,
                        fieldValue(boundary_grad_temperature_result, 0, qp, d));
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D: time transient incompressible rotating wall - steady
template<class EvalType>
void testTimeTransientIncompressibleRotatingWallSteady2D(
    const bool build_temp_equ)
{
    const Kokkos::Array<double, 3> time_values = {-0.5, -3.0, 3.0};
    const Kokkos::Array<double, 1> init_values = {-1.0};
    const Kokkos::Array<double, 2> exp_fields = {-1.955, 1.475};

    testEval<EvalType, 2>(time_values, init_values, exp_fields, build_temp_equ);
}

// 2-D: time transient incompressible rotating wall residual
TEST(TimeTransientIncompressibleRotatingWallSteady2D, residual)
{
    testTimeTransientIncompressibleRotatingWallSteady2D<panzer::Traits::Residual>(
        false);
}

// 2-D: time transient incompressible rotating wall jacobian
TEST(TimeTransientIncompressibleRotatingWallSteady2D, jacobian)
{
    testTimeTransientIncompressibleRotatingWallSteady2D<panzer::Traits::Jacobian>(
        false);
}

//---------------------------------------------------------------------------//
// 2-D: time transient incompressible rotating wall - time > time_final
template<class EvalType>
void testTimeTransientIncompressibleRotatingWallTimeFinal2D(
    const bool build_temp_equ)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 3.5};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 2> exp_fields = {-1.955, 1.475};

    testEval<EvalType, 2>(time_values, init_values, exp_fields, build_temp_equ);
}

// 2-D: time transient incompressible rotating wall residual
TEST(TimeTransientIncompressibleRotatingWallTimeFinal2D, residual)
{
    testTimeTransientIncompressibleRotatingWallTimeFinal2D<
        panzer::Traits::Residual>(false);
}

// 2-D: time transient incompressible rotating wall jacobian
TEST(TimeTransientIncompressibleRotatingWallTimeFinal2D, jacobian)
{
    testTimeTransientIncompressibleRotatingWallTimeFinal2D<
        panzer::Traits::Jacobian>(false);
}

//---------------------------------------------------------------------------//
// 2-D: time transient incompressible rotating wall - time < time_init
template<class EvalType>
void testTimeTransientIncompressibleRotatingWallTimeInit2D(
    const bool build_temp_equ)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 0.2};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 2> exp_fields = {-0.9775, 0.7375};

    testEval<EvalType, 2>(time_values, init_values, exp_fields, build_temp_equ);
}

// 2-D: time transient incompressible rotating wall residual
TEST(TimeTransientIncompressibleRotatingWallTimeInit2D, residual)
{
    testTimeTransientIncompressibleRotatingWallTimeInit2D<panzer::Traits::Residual>(
        false);
}

// 2-D: time-transient incompressible rotating wall jacobian
TEST(TimeTransientIncompressibleRotatingWallTimeInit2D, jacobian)
{
    testTimeTransientIncompressibleRotatingWallTimeInit2D<panzer::Traits::Jacobian>(
        false);
}

//---------------------------------------------------------------------------//
// 2-D: time transient incompressible rotating wall - time_init < time <
// time_final
template<class EvalType>
void testTimeTransientIncompressibleRotatingWallTimeIntermediate2D(
    const bool build_temp_equ)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 1.5};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 2> exp_fields = {-1.3685, 1.0325};

    testEval<EvalType, 2>(time_values, init_values, exp_fields, build_temp_equ);
}

// 2-D: time transient incompressible rotating wall residual
TEST(TimeTransientIncompressibleRotatingWallTimeIntermediate2D, residual)
{
    testTimeTransientIncompressibleRotatingWallTimeIntermediate2D<
        panzer::Traits::Residual>(false);
}

// 2-D: time transient incompressible rotating wall jacobian
TEST(TimeTransientIncompressibleRotatingWallTimeIntermediate2D, jacobian)
{
    testTimeTransientIncompressibleRotatingWallTimeIntermediate2D<
        panzer::Traits::Jacobian>(false);
}

//---------------------------------------------------------------------------//
// 3-D: time transient incompressible rotating wall - time_init < time <
// time_final
template<class EvalType>
void testTimeTransientIncompressibleRotatingWallTimeIntermediate3D(
    const bool build_temp_equ)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 1.5};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 3> exp_fields = {-1.3685, 1.0325, 0.0};

    testEval<EvalType, 3>(time_values, init_values, exp_fields, build_temp_equ);
}

// 3-D: time transient incompressible rotating wall residual
TEST(TimeTransientIncompressibleRotatingWallTimeIntermediate3D, residual)
{
    testTimeTransientIncompressibleRotatingWallTimeIntermediate3D<
        panzer::Traits::Residual>(false);
}

// 3-D: time transient incompressible rotating wall jacobian
TEST(TimeTransientIncompressibleRotatingWallTimeIntermediate3D, jacobian)
{
    testTimeTransientIncompressibleRotatingWallTimeIntermediate3D<
        panzer::Traits::Jacobian>(false);
}
//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD

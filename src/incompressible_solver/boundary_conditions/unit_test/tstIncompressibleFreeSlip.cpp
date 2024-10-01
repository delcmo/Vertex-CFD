#include <VertexCFD_EvaluatorTestHarness.hpp>

#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleFreeSlip.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

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

    double _phi, _u_0, _u_1, _u_2;
    bool _build_temp_equ;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _temperature;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double phi,
                 const double u_0,
                 const double u_1,
                 const double u_2,
                 const bool build_temp_equ)
        : _phi(phi)
        , _u_0(u_0)
        , _u_1(u_1)
        , _u_2(u_2)
        , _build_temp_equ(build_temp_equ)
        , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , _temperature("temperature", ir.dl_scalar)
        , _velocity_0("velocity_0", ir.dl_scalar)
        , _velocity_1("velocity_1", ir.dl_scalar)
        , _velocity_2("velocity_2", ir.dl_scalar)
        , _normals("Side Normal", ir.dl_vector)
        , _grad_velocity_0("GRAD_velocity_0", ir.dl_vector)
        , _grad_velocity_1("GRAD_velocity_1", ir.dl_vector)
        , _grad_velocity_2("GRAD_velocity_2", ir.dl_vector)
        , _grad_temperature("GRAD_temperature", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        if (_build_temp_equ)
            this->addEvaluatedField(_temperature);
        this->addEvaluatedField(_velocity_0);
        this->addEvaluatedField(_velocity_1);
        this->addEvaluatedField(_velocity_2);

        this->addEvaluatedField(_normals);

        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);
        if (_build_temp_equ)
            this->addEvaluatedField(_grad_temperature);

        this->setName("Incompressible FreeSlip Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        // Set scalar variables
        _lagrange_pressure.deep_copy(_phi);
        _velocity_0.deep_copy(_u_0);
        _velocity_1.deep_copy(_u_1);
        _velocity_2.deep_copy(_u_2);
        if (_build_temp_equ)
            _temperature.deep_copy(_u_0 + _u_1);

        Kokkos::parallel_for(
            "incompressible free slip test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int c) const
    {
        const int num_point = _lagrange_pressure.extent(1);
        const int num_space_dim = _grad_velocity_0.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            // Set gradient and normal vectors
            for (int d = 0; d < num_space_dim; ++d)
            {
                _normals(c, qp, d) = 0.01 * (d + 1) * (d * num_point + 1)
                                     * (qp + 1);
                int dqp = (qp + 1) * (d + num_point + 1);
                _grad_velocity_0(c, qp, d) = _u_0 * dqp;
                _grad_velocity_1(c, qp, d) = _u_1 * dqp;
                _grad_velocity_2(c, qp, d) = _u_2 * dqp;
                if (_build_temp_equ)
                    _grad_temperature(c, qp, d) = (_u_0 + _u_1) * dqp;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool build_temp_equ)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Initialize values and create dependencies
    const double phi = 1.5;
    const double u_0 = 2.0;
    const double u_1 = 2.5;
    const double u_2
        = num_space_dim == 3 ? 2.75 : std::numeric_limits<double>::quiet_NaN();
    double vel[3] = {u_0, u_1, u_2};
    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(
        *test_fixture.ir, phi, u_0, u_1, u_2, build_temp_equ));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Equation of state
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

    // Create free slip evaluator.
    auto free_slip_eval = Teuchos::rcp(
        new BoundaryCondition::
            IncompressibleFreeSlip<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir, fluid_prop));
    test_fixture.registerEvaluator<EvalType>(free_slip_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        free_slip_eval->_boundary_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            free_slip_eval->_boundary_velocity[dim]);
        test_fixture.registerTestField<EvalType>(
            free_slip_eval->_boundary_grad_velocity[dim]);
    }

    // Evaluate incompressible free slip
    test_fixture.evaluate<EvalType>();

    // Get free slip field
    auto boundary_lagrange_pressure_result
        = test_fixture.getTestFieldData<EvalType>(
            free_slip_eval->_boundary_lagrange_pressure);

    // Loop over quadrature points and mesh dimension
    const int num_point = boundary_lagrange_pressure_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        // Lagrange pressure
        EXPECT_DOUBLE_EQ(phi,
                         fieldValue(boundary_lagrange_pressure_result, 0, qp));

        // Calculate velocity boundary using normal vector
        const auto normals
            = test_fixture.getTestFieldData<EvalType>(dep_eval->_normals);
        const double n0 = fieldValue(normals, 0, qp, 0);
        const double n1 = fieldValue(normals, 0, qp, 1);
        const double n2 = num_space_dim == 3
                              ? fieldValue(normals, 0, qp, 2)
                              : std::numeric_limits<double>::quiet_NaN();

        double u_dot_n = u_0 * n0 + u_1 * n1;
        if (num_space_dim == 3)
            u_dot_n += u_2 * n2;
        const double u_bnd = u_0 - u_dot_n * n0;
        const double v_bnd = u_1 - u_dot_n * n1;
        const double w_bnd = num_space_dim == 3 ? u_2 - u_dot_n * n2 : 0.0;
        double vel_bnd[3] = {u_bnd, v_bnd, w_bnd};

        double grad_T_dot_n = 0.0;
        if (build_temp_equ)
        {
            for (int d = 0; d < num_space_dim; ++d)
            {
                const int dqp = (qp + 1) * (d + num_point + 1);
                grad_T_dot_n += (u_0 + u_1) * dqp
                                * fieldValue(normals, 0, qp, d);
            }
        }

        // Temperature
        if (build_temp_equ)
        {
            const auto boundary_temperature_result
                = test_fixture.getTestFieldData<EvalType>(
                    free_slip_eval->_boundary_temperature);
            EXPECT_DOUBLE_EQ(u_0 + u_1,
                             fieldValue(boundary_temperature_result, 0, qp));
        }

        // Loop over mesh dimension to assert gradient vectors
        for (int d = 0; d < num_space_dim; ++d)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    free_slip_eval->_boundary_velocity[d]);
            EXPECT_DOUBLE_EQ(vel_bnd[d],
                             fieldValue(boundary_velocity_d_result, 0, qp));

            const int dqp = (qp + 1) * (d + num_point + 1);

            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                const auto boundary_grad_velocity_d_result
                    = test_fixture.getTestFieldData<EvalType>(
                        free_slip_eval->_boundary_grad_velocity[vel_dim]);
                EXPECT_DOUBLE_EQ(
                    vel[vel_dim] * dqp,
                    fieldValue(boundary_grad_velocity_d_result, 0, qp, d));
            }
            if (build_temp_equ)
            {
                const auto grad_temp_result
                    = test_fixture.getTestFieldData<EvalType>(
                        dep_eval->_grad_temperature);
                const double grad_temp_ref
                    = fieldValue(grad_temp_result, 0, qp, d)
                      - grad_T_dot_n * fieldValue(normals, 0, qp, d);
                const auto boundary_grad_temperature_result
                    = test_fixture.getTestFieldData<EvalType>(
                        free_slip_eval->_boundary_grad_temperature);
                EXPECT_DOUBLE_EQ(
                    grad_temp_ref,
                    fieldValue(boundary_grad_temperature_result, 0, qp, d));
            }
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D incompressible isothermal freeSlip
TEST(IncompressibleFreeSlipIsothermal2D, residual)
{
    testEval<panzer::Traits::Residual, 2>(false);
}

TEST(IncompressibleFreeSlipIsothermal2D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false);
}

//---------------------------------------------------------------------------//
// 3-D incompressible isothermal freeSlip
TEST(IncompressibleFreeSlipIsothermal3D, residual)
{
    testEval<panzer::Traits::Residual, 3>(false);
}

TEST(IncompressibleFreeSlipIsothermal3D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false);
}

//---------------------------------------------------------------------------//
// 2-D incompressible freeSlip
TEST(IncompressibleFreeSlip2D, residual)
{
    testEval<panzer::Traits::Residual, 2>(true);
}

TEST(IncompressibleFreeSlip2D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true);
}

//---------------------------------------------------------------------------//
// 3-D incompressible freeSlip
TEST(IncompressibleFreeSlip3D, residual)
{
    testEval<panzer::Traits::Residual, 3>(true);
}

TEST(IncompressibleFreeSlip3D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true);
}
//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD

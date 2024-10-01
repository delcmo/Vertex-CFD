#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_ElectricPotentialFixed.hpp"
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
// Input type
enum class InputType
{
    steady,
    pastFinalTime,
    transient,
    preInitialTime
};

//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _electric_potential;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_electric_potential;

    Dependencies(const panzer::IntegrationRule& ir)
        : _electric_potential("electric_potential", ir.dl_scalar)
        , _grad_electric_potential("GRAD_electric_potential", ir.dl_vector)
    {
        this->addEvaluatedField(_electric_potential);
        this->addEvaluatedField(_grad_electric_potential);
        this->setName(
            "Time Transient Electric Potential Fixed Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _electric_potential.deep_copy(2.0);

        Kokkos::parallel_for(
            "time transient electric potential fixed test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _grad_electric_potential.extent(1);
        const int num_grad_dim = _grad_electric_potential.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int d = 0; d < num_grad_dim; ++d)
            {
                const int dqp = (qp + 1) * (d + num_point + 1);
                _grad_electric_potential(c, qp, d) = dqp * 3.0;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_grad_dim, const InputType input_type)
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create the param list to initialize the evaluator
    Teuchos::ParameterList bc_params;
    double time = 0.0;
    double exp_value = 0.0;
    switch (input_type)
    {
        case (InputType::steady):
            exp_value = 2.0;
            bc_params.set("Final Value", exp_value);
            break;
        case (InputType::pastFinalTime):
            exp_value = 2.0;
            bc_params.set("Final Value", exp_value);
            bc_params.set("Initial Value", 0.5);
            time = 1.5;
            bc_params.set("Time Final", 1.0);
            bc_params.set("Time Initial", 0.1);
            break;
        case (InputType::transient):
            exp_value = 3.5;
            bc_params.set("Final Value", 3.0);
            bc_params.set("Initial Value", 4.0);
            time = 1.5;
            bc_params.set("Time Final", 2.0);
            bc_params.set("Time Initial", 1.0);
            break;
        case (InputType::preInitialTime):
            exp_value = 3.0;
            bc_params.set("Final Value", 4.0);
            bc_params.set("Initial Value", exp_value);
            time = 0.5;
            bc_params.set("Time Final", 2.0);
            bc_params.set("Time Initial", 1.0);
            break;
    }

    // Create fixed evaluator.
    auto fixed_eval = Teuchos::rcp(
        new BoundaryCondition::ElectricPotentialFixed<EvalType, panzer::Traits>(
            *test_fixture.ir, bc_params));
    test_fixture.registerEvaluator<EvalType>(fixed_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        fixed_eval->_boundary_electric_potential);
    test_fixture.registerTestField<EvalType>(
        fixed_eval->_boundary_grad_electric_potential);

    // Set time
    test_fixture.setTime(time);

    // Evaluate time transient fixed BC.
    test_fixture.evaluate<EvalType>();

    // Check the time transient fixed BC.
    const auto boundary_ep_result = test_fixture.getTestFieldData<EvalType>(
        fixed_eval->_boundary_electric_potential);
    const auto boundary_grad_ep_result
        = test_fixture.getTestFieldData<EvalType>(
            fixed_eval->_boundary_grad_electric_potential);

    const int num_point = boundary_ep_result.extent(1);

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_value, fieldValue(boundary_ep_result, 0, qp));

        for (int d = 0; d < num_grad_dim; ++d)
        {
            const int dqp = (qp + 1) * (d + num_point + 1);
            EXPECT_DOUBLE_EQ(dqp * 3.0,
                             fieldValue(boundary_grad_ep_result, 0, qp, d));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D: time transient fixed - steady
template<class EvalType>
void testTimeTransientElectricPotentialFixedSteady2D()
{
    testEval<EvalType>(2, InputType::steady);
}

// 2-D time transient fixed residual
TEST(TimeTransientElectricPotentialFixedSteady2D, residual)
{
    testTimeTransientElectricPotentialFixedSteady2D<panzer::Traits::Residual>();
}

// 2-D time transient fixed jacobian
TEST(TimeTransientElectricPotentialFixedSteady2D, jacobian)
{
    testTimeTransientElectricPotentialFixedSteady2D<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// 2-D time transient fixed - time > time_final
template<class EvalType>
void testTimeTransientElectricPotentialFixedTimeFinal2D()
{
    testEval<EvalType>(2, InputType::pastFinalTime);
}

// 2-D time transient fixed residual
TEST(TimeTransientElectricPotentialFixedTimeFinal2D, residual)
{
    testTimeTransientElectricPotentialFixedTimeFinal2D<panzer::Traits::Residual>();
}

// 2-D time transient fixed jacobian
TEST(TimeTransientElectricPotentialFixedTimeFinal2D, jacobian)
{
    testTimeTransientElectricPotentialFixedTimeFinal2D<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// 2-D time transient fixed - time < time_init
template<class EvalType>
void testTimeTransientElectricPotentialFixedTimeInit2D()
{
    testEval<EvalType>(2, InputType::preInitialTime);
}

// 2-D time transient fixed residual
TEST(TimeTransientElectricPotentialFixedTimeInit2D, residual)
{
    testTimeTransientElectricPotentialFixedTimeInit2D<panzer::Traits::Residual>();
}

// 2-D time transient fixed jacobian
TEST(TimeTransientElectricPotentialFixedTimeInit2D, jacobian)
{
    testTimeTransientElectricPotentialFixedTimeInit2D<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// 2-D time transient fixed - time_init < time < time_final. The expected
// values should be the average value between the final and the initial values.
template<class EvalType>
void testTimeTransientElectricPotentialFixedTimeIntermediate2D()
{
    testEval<EvalType>(2, InputType::transient);
}

// 2-D time transient fixed residual
TEST(TimeTransientElectricPotentialFixedTimeIntermediate2D, residual)
{
    testTimeTransientElectricPotentialFixedTimeIntermediate2D<
        panzer::Traits::Residual>();
}

// 2-D time transient fixed jacobian
TEST(TimeTransientElectricPotentialFixedTimeIntermediate2D, jacobian)
{
    testTimeTransientElectricPotentialFixedTimeIntermediate2D<
        panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// 3-D time transient fixed - time_init < time < time_final. The expected
// values should be the average value between the final and the initial values
template<class EvalType>
void testTimeTransientElectricPotentialFixedTimeIntermediate3D()
{
    testEval<EvalType>(3, InputType::transient);
}

// 3-D time transient fixed residual
TEST(TimeTransientElectricPotentialFixedTimeIntermediate3D, residual)
{
    testTimeTransientElectricPotentialFixedTimeIntermediate3D<
        panzer::Traits::Residual>();
}

// 3-D time transient fixed jacobian
TEST(TimeTransientElectricPotentialFixedTimeIntermediate3D, jacobian)
{
    testTimeTransientElectricPotentialFixedTimeIntermediate2D<
        panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD

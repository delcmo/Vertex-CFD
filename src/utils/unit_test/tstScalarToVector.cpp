#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <VertexCFD_Utils_ScalarToVector.hpp>
#include <VertexCFD_Utils_VelocityDim.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

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
template<typename EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;
    int num_vel_dim;

    // Array of field solutions
    std::vector<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>> velocities;
    std::vector<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>>
        dxdt_velocities;
    std::vector<PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>>
        grad_velocities;

    Dependencies(const panzer::IntegrationRule& ir, int dim)
        : num_vel_dim(dim)
    {
        velocities.resize(num_vel_dim);
        dxdt_velocities.resize(num_vel_dim);
        grad_velocities.resize(num_vel_dim);
        std::string name;
        for (int dim = 0; dim < num_vel_dim; ++dim)
        {
            name = "velocity_" + std::to_string(dim);
            velocities[dim]
                = PHX::MDField<scalar_type, panzer::Cell, panzer::Point>(
                    name, ir.dl_scalar);

            name = "DXDT_velocity_" + std::to_string(dim);
            dxdt_velocities[dim]
                = PHX::MDField<scalar_type, panzer::Cell, panzer::Point>(
                    name, ir.dl_scalar);

            name = "GRAD_velocity_" + std::to_string(dim);
            grad_velocities[dim] = PHX::
                MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>(
                    name, ir.dl_vector);

            this->addEvaluatedField(velocities[dim]);
            this->addEvaluatedField(dxdt_velocities[dim]);
            this->addEvaluatedField(grad_velocities[dim]);
        }

        this->setName("ScalarToVector Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData workset) override
    {
        const int num_vel_dim = velocities.size();
        const int num_points = velocities[0].extent(1);

        for (int vel_dim = 0; vel_dim < num_vel_dim; ++vel_dim)
        {
            // Create host mirrors of fields
            auto vel_view = velocities[vel_dim].get_view();
            auto dxdt_vel_view = dxdt_velocities[vel_dim].get_view();
            auto grad_vel_view = grad_velocities[vel_dim].get_view();
            auto host_vel = Kokkos::create_mirror(vel_view);
            auto host_dxdt_vel = Kokkos::create_mirror(dxdt_vel_view);
            auto host_grad_vel = Kokkos::create_mirror(grad_vel_view);

            for (int cell = 0; cell < workset.num_cells; ++cell)
            {
                for (int point = 0; point < num_points; ++point)
                {
                    // Set velocity values
                    double val = static_cast<double>(vel_dim + 3 * point);
                    host_vel(cell, point) = val;

                    // Set dxdt_velocity values
                    val = static_cast<double>(100 + vel_dim + 3 * point);
                    host_dxdt_vel(cell, point) = val;

                    // Set grad_velocity values
                    int num_space_dim = grad_vel_view.extent(2);
                    for (int space_dim = 0; space_dim < num_space_dim;
                         ++space_dim)
                    {
                        val = static_cast<double>(vel_dim + 3 * point
                                                  + 9 * space_dim);
                        host_grad_vel(cell, point, space_dim) = val;
                    }
                }
            }
            Kokkos::deep_copy(vel_view, host_vel);
            Kokkos::deep_copy(dxdt_vel_view, host_dxdt_vel);
            Kokkos::deep_copy(grad_vel_view, host_grad_vel);
        }
    }
};

template<typename EvalType, int NumVelDim, int NumSpaceDim>
void testEval()
{
    // Setup test fixture.
    constexpr int num_vel_dim = NumVelDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_vel_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Create test dependency to set initial fields
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, num_vel_dim));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Create evaluator.
    auto eval = Teuchos::rcp(new Utils::ScalarToVector<EvalType, VelocityDim>(
        ir, "velocity", num_vel_dim, true));
    test_fixture.registerEvaluator<EvalType>(eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(eval->_vector_fields);
    test_fixture.registerTestField<EvalType>(eval->_vector_dxdt_fields);
    test_fixture.registerTestField<EvalType>(eval->_vector_grad_fields);

    // Evaluate
    test_fixture.evaluate<EvalType>();

    // Check the values
    const auto vector_vel
        = test_fixture.getTestFieldData<EvalType>(eval->_vector_fields);
    const auto vector_dxdt_vel
        = test_fixture.getTestFieldData<EvalType>(eval->_vector_dxdt_fields);
    const auto vector_grad_vel
        = test_fixture.getTestFieldData<EvalType>(eval->_vector_grad_fields);

    // Check the solution
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int vel_dim = 0; vel_dim < num_vel_dim; ++vel_dim)
        {
            // Test velocity
            double expected = static_cast<double>(vel_dim + 3 * qp);
            EXPECT_DOUBLE_EQ(expected, fieldValue(vector_vel, 0, qp, vel_dim));

            // Test dxdt_velocity
            expected = static_cast<double>(100 + vel_dim + 3 * qp);
            EXPECT_DOUBLE_EQ(expected,
                             fieldValue(vector_dxdt_vel, 0, qp, vel_dim));

            // Test grad_velocity
            for (int space_dim = 0; space_dim < NumSpaceDim; ++space_dim)
            {
                expected
                    = static_cast<double>(vel_dim + 3 * qp + 9 * space_dim);
                EXPECT_DOUBLE_EQ(
                    expected,
                    fieldValue(vector_grad_vel, 0, qp, space_dim, vel_dim));
            }
        }
    }
}

//---------------------------------------------------------------------------//
TEST(S2V_22, residual_test)
{
    testEval<panzer::Traits::Residual, 2, 2>();
}

//---------------------------------------------------------------------------//
TEST(S2V_22, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2, 2>();
}

//---------------------------------------------------------------------------//
TEST(S2V_33, residual_test)
{
    testEval<panzer::Traits::Residual, 3, 3>();
}

//---------------------------------------------------------------------------//
TEST(S2V_33, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3, 3>();
}

//---------------------------------------------------------------------------//
TEST(S2V_32, residual_test)
{
    testEval<panzer::Traits::Residual, 3, 2>();
}

//---------------------------------------------------------------------------//
TEST(S2V_32, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3, 2>();
}

} // end namespace Test
} // end namespace VertexCFD

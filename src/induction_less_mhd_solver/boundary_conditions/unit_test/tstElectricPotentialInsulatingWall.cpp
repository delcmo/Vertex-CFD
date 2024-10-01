#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_ElectricPotentialInsulatingWall.hpp"
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _electric_potential;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_electric_potential;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    Dependencies(const panzer::IntegrationRule& ir)
        : _electric_potential("electric_potential", ir.dl_scalar)
        , _grad_electric_potential("GRAD_electric_potential", ir.dl_vector)
        , _normals("Side Normal", ir.dl_vector)
    {
        this->addEvaluatedField(_electric_potential);
        this->addEvaluatedField(_grad_electric_potential);
        this->addEvaluatedField(_normals);
        this->setName(
            "Electric Potential Insulating Wall Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _electric_potential.deep_copy(2.0);

        Kokkos::parallel_for(
            "electric potential insulating wall test dependencies",
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
                const int dqp = (qp + 1) * (d + 1);
                _grad_electric_potential(c, qp, d) = dqp * 3.0;
                _normals(c, qp, d) = dqp * 0.1;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_grad_dim)
{
    // Test fixture
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create evaluator.
    const auto fixed_eval = Teuchos::rcp(
        new BoundaryCondition::ElectricPotentialInsulatingWall<EvalType,
                                                               panzer::Traits>(
            *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(fixed_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        fixed_eval->_boundary_electric_potential);
    test_fixture.registerTestField<EvalType>(
        fixed_eval->_boundary_grad_electric_potential);

    // Evaluate variables.
    test_fixture.evaluate<EvalType>();

    // Check values.
    const auto boundary_ep_result = test_fixture.getTestFieldData<EvalType>(
        fixed_eval->_boundary_electric_potential);
    const auto boundary_grad_ep_result
        = test_fixture.getTestFieldData<EvalType>(
            fixed_eval->_boundary_grad_electric_potential);

    const int num_point = boundary_ep_result.extent(1);
    const double ep_dot_n = num_grad_dim == 2 ? 1.5 : 4.2;

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(2.0, fieldValue(boundary_ep_result, 0, qp));

        for (int d = 0; d < num_grad_dim; ++d)
        {
            const int dqp = (qp + 1) * (d + 1);
            EXPECT_DOUBLE_EQ(dqp * 3.0 - ep_dot_n * dqp * 0.1,
                             fieldValue(boundary_grad_ep_result, 0, qp, d));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D residual
TEST(InsulatingWall2D, residual)
{
    testEval<panzer::Traits::Residual>(2);
}

// 2-D jacobian
TEST(InsulatingWall2D, jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//---------------------------------------------------------------------------//
// 3-D residual
TEST(InsulatingWall3D, residual)
{
    testEval<panzer::Traits::Residual>(3);
}

// 3-D jacobian
TEST(InsulatingWall3D, jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD

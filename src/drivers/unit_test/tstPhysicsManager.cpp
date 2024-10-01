#include <VertexCFD_DriverUnitTestConfig.hpp>
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <drivers/VertexCFD_InitialConditionManager.hpp>
#include <drivers/VertexCFD_MeshManager.hpp>
#include <drivers/VertexCFD_PhysicsManager.hpp>
#include <parameters/VertexCFD_ParameterDatabase.hpp>

#include <Panzer_Traits.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
auto createPhysicsManager(const std::string& location,
                          const std::string& filename)
{
    // Initialize mesh dimension
    constexpr int num_space_dim = NumSpaceDim;

    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    int argc = 2;
    const std::string option = "--i=";
    std::string argv_str = option + location + filename;
    char* argv[2];
    argv[1] = &argv_str[0];

    // Setup database.
    auto parameter_db
        = Teuchos::rcp(new Parameter::ParameterDatabase(comm, argc, argv));

    // Create the mesh.
    auto mesh_manager = Teuchos::rcp(new MeshManager(*parameter_db, comm));

    // Create physics.
    double t_init = 1.3;
    auto physics_manager = Teuchos::rcp(
        new PhysicsManager(std::integral_constant<int, num_space_dim>{},
                           parameter_db,
                           mesh_manager,
                           t_init));

    // Finish physics.
    physics_manager->setupModel();

    return physics_manager;
}

//---------------------------------------------------------------------------//
TEST(PhysicsManager, manager_test)
{
    auto physics_manager = createPhysicsManager<2>(
        VERTEXCFD_DRIVER_TEST_INPUT_DIR, "simple_box_2d.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(4, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test boundary factory model of incompressible NS equations with full
// induction MHD model enabled using a dummy input file for a 3D geometry.
TEST(PhysicsManagerFIM, manager_test)
{
    auto physics_manager = createPhysicsManager<3>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_fim_3d.xml");
    // Check bouondary logic.
    EXPECT_EQ(6, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
}

//---------------------------------------------------------------------------//
template<class EvalType>
void testScalarParameter()
{
    // Create physics manager.
    auto physics_manager = createPhysicsManager<2>(
        VERTEXCFD_DRIVER_TEST_INPUT_DIR, "simple_box_2d.xml");

    // Add a scalar parameter.
    const std::string scalar_name = "scalar_param";
    const double param_value = 2.03;
    auto param_id
        = physics_manager->addScalarParameter(scalar_name, param_value);
    EXPECT_EQ(0, param_id);
    EXPECT_EQ(0, physics_manager->getParameterIndex(scalar_name));

    // Finish physics.
    physics_manager->setupModel();

    // Check scalar parameter.
    auto param_lib = physics_manager->globalData()->pl;
    EXPECT_EQ(param_value, param_lib->getValue<EvalType>(scalar_name));
}
//---------------------------------------------------------------------------//
TEST(PhysicsManager, scalar_parameter_test_residual)
{
    testScalarParameter<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(PhysicsManager, scalar_parameter_test_jacobian)
{
    testScalarParameter<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD

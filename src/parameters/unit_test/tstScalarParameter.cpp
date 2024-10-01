#include "VertexCFD_ParameterUnitTestConfig.hpp"

#include "parameters/VertexCFD_ParameterDatabase.hpp"
#include "parameters/VertexCFD_ScalarParameter.hpp"

#include "drivers/VertexCFD_InitialConditionManager.hpp"
#include "drivers/VertexCFD_MeshManager.hpp"
#include "drivers/VertexCFD_PhysicsManager.hpp"

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
double getValue(const panzer::Traits::Residual::ScalarT& observed)
{
    return observed;
}

//---------------------------------------------------------------------------//
double getValue(const panzer::Traits::Jacobian::ScalarT& observed)
{
    return observed.val();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void testScalarParameter()
{
    // Setup base data structures to make a parameter library.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());
    const std::string location = VERTEXCFD_RESPONSE_TEST_DATA_DIR;
    const std::string file = "response_manager_test.xml";
    auto parameter_db = Teuchos::rcp(
        new Parameter::ParameterDatabase(comm, location + file));
    parameter_db->physicsParameters()
        ->sublist("FluidPhysicsBlock", true)
        .sublist("Data", true)
        .set("Basis Order", 1);
    auto mesh_manager = Teuchos::rcp(new MeshManager(*parameter_db, comm));
    auto physics_manager = Teuchos::rcp(new PhysicsManager(
        std::integral_constant<int, 2>{}, parameter_db, mesh_manager));
    physics_manager->setupModel();

    // Add a global parameter
    physics_manager->addScalarParameter("Global Parameter", 1.234);

    // Make a parameter to change.
    typename EvalType::ScalarT param_value = 2.345;

    // Make global parameter.
    auto global_param = Teuchos::rcp(new Parameter::ScalarParameter<EvalType>(
        "Global Parameter", param_value));
    EXPECT_EQ("Global Parameter", global_param->name());
    EXPECT_EQ(2.345, getValue(param_value));
    global_param->update(*(physics_manager->globalData()));
    EXPECT_EQ(1.234, getValue(param_value));
}

//---------------------------------------------------------------------------//
TEST(ScalarParameter, residual)
{
    testScalarParameter<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(ScalarParameter, jacobian)
{
    testScalarParameter<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD

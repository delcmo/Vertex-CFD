#include <VertexCFD_ParameterUnitTestConfig.hpp>

#include <parameters/VertexCFD_ParameterDatabase.hpp>

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
void testDefaultDatabase()
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Create database.
    VertexCFD::Parameter::ParameterDatabase parameter_db(comm);

    // Check that all parameter lists got populated.
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.meshParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.physicsParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.blockMappingParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.scalarParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.generalScalarParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.boundaryConditionParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.initialConditionParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.closureModelParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.responseOutputParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.userParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.outputParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.readRestartParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.writeRestartParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.writeMatrixParameters()));
    EXPECT_TRUE(Teuchos::is_null(parameter_db.profilingParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.transientSolverParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.linearSolverParameters()));

    // Check that we added the communicator to the user data.
    auto param_comm
        = parameter_db.userParameters()
              ->get<Teuchos::RCP<const Teuchos::Comm<int>>>("Comm");
    EXPECT_EQ(comm->getRank(), param_comm->getRank());
    EXPECT_EQ(comm->getSize(), param_comm->getSize());
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, default_test)
{
    testDefaultDatabase();
}

//---------------------------------------------------------------------------//
void testInputParser(const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
                     const VertexCFD::Parameter::ParameterDatabase& parameter_db)
{
    // Check that all parameter lists got populated.
    EXPECT_DOUBLE_EQ(1.0, parameter_db.meshParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(1.4,
                     parameter_db.physicsParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        1.3, parameter_db.blockMappingParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        1.6, parameter_db.boundaryConditionParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        1.7, parameter_db.initialConditionParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        1.8, parameter_db.closureModelParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.4, parameter_db.responseOutputParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(1.5, parameter_db.userParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(1.1,
                     parameter_db.outputParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.3, parameter_db.readRestartParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.2, parameter_db.writeRestartParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.1, parameter_db.writeMatrixParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(1.2,
                     parameter_db.profilingParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.0, parameter_db.transientSolverParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        1.9, parameter_db.linearSolverParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(2.5,
                     parameter_db.scalarParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.6, parameter_db.generalScalarParameters()->get<double>("value"));

    // Check the parameter communicator.
    EXPECT_EQ(comm->getRank(), parameter_db.comm()->getRank());
    EXPECT_EQ(comm->getSize(), parameter_db.comm()->getSize());

    // Check that we added the communicator to the user data.
    auto param_comm
        = parameter_db.userParameters()
              ->get<Teuchos::RCP<const Teuchos::Comm<int>>>("Comm");
    EXPECT_EQ(comm->getRank(), param_comm->getRank());
    EXPECT_EQ(comm->getSize(), param_comm->getSize());
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, argv_test)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    int argc = 2;
    std::string option = "--i=";
    std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    std::string file = "input_parser_test.xml";
    std::string argv_str = option + location + file;
    char* argv[2];
    argv[1] = &argv_str[0];
    VertexCFD::Parameter::ParameterDatabase parameter_db(comm, argc, argv);

    // Test
    testInputParser(comm, parameter_db);
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, file_test)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    std::string file = "input_parser_test.xml";
    std::string filename = location + file;
    VertexCFD::Parameter::ParameterDatabase parameter_db(comm, filename);

    // Test
    testInputParser(comm, parameter_db);
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, list_test)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    std::string file = "input_parser_test.xml";
    std::string filename = location + file;
    VertexCFD::Parameter::ParameterDatabase file_db(comm, filename);

    // Create a new parameter database from the input list.
    VertexCFD::Parameter::ParameterDatabase parameter_db(
        comm, file_db.allParameters());

    // Test
    testInputParser(comm, parameter_db);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD

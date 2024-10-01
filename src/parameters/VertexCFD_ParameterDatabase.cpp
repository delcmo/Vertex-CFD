#include "VertexCFD_ParameterDatabase.hpp"
#include "VertexCFD_GeneralScalarParameterInput.hpp"
#include "VertexCFD_ScalarParameterInput.hpp"

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterEntryXMLConverterDB.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
ParameterDatabase::ParameterDatabase(
    const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm)
    : _comm(comm)
{
    // Create sublists.
    _mesh_params = Teuchos::parameterList();
    _physics_params = Teuchos::parameterList();
    _scalar_params = Teuchos::parameterList();
    _general_scalar_params = Teuchos::parameterList();
    _block_mapping_params = Teuchos::parameterList();
    _bc_params = Teuchos::parameterList();
    _ic_params = Teuchos::parameterList();
    _closure_params = Teuchos::parameterList();
    _response_output_params = Teuchos::parameterList();
    _user_params = Teuchos::parameterList();
    _output_params = Teuchos::parameterList();
    _read_restart_params = Teuchos::parameterList();
    _write_restart_params = Teuchos::parameterList();
    _write_matrix_params = Teuchos::parameterList();
    _transient_solver_params = Teuchos::parameterList();
    _linear_solver_params = Teuchos::parameterList();

    // Store the MPI communicator in the "User Data" ParameterList.
    _user_params->set<Teuchos::RCP<const Teuchos::Comm<int>>>("Comm", _comm);
}

//---------------------------------------------------------------------------//
ParameterDatabase::ParameterDatabase(
    const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
    const Teuchos::RCP<Teuchos::ParameterList>& parameters)
    : _comm(comm)
    , _input_params(parameters)
{
    // Get the sublists.
    extractSublists();
}

//---------------------------------------------------------------------------//
ParameterDatabase::ParameterDatabase(
    const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
    const std::string& xml_file)
    : _comm(comm)
{
    // Parse file.
    readParameterFile(xml_file);
}

//---------------------------------------------------------------------------//
ParameterDatabase::ParameterDatabase(
    const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
    int argc,
    char* argv[])
    : _comm(comm)
{
    // Get the input file from the command line arguments.
    Teuchos::CommandLineProcessor clp;
    std::string input_file = "input.xml";
    clp.setOption("i", &input_file, "XML Input File");
    clp.parse(argc, argv, &std::cerr);

    // Parse file.
    readParameterFile(input_file);
}

//---------------------------------------------------------------------------//
void ParameterDatabase::readParameterFile(const std::string& xml_file)
{
    // Add custom input types.
    TEUCHOS_ADD_TYPE_CONVERTER(ScalarParameterInput);
    TEUCHOS_ADD_TYPE_CONVERTER(GeneralScalarParameterInput);

    // Build a parameter list from the inputs
    _input_params = Teuchos::parameterList("Input Parameters");
    Teuchos::updateParametersFromXmlFileAndBroadcast(
        xml_file, _input_params.ptr(), *_comm);

    // Get the sublists.
    extractSublists();
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::requiredSublist(const std::string& name)
{
    if (!_input_params->isSublist(name))
    {
        std::string msg = "Sublist " + name + " missing from input";
        throw std::logic_error(msg);
    }
    return Teuchos::parameterList(_input_params->sublist(name));
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::optionalSublist(const std::string& name)
{
    if (_input_params->isSublist(name))
    {
        return Teuchos::parameterList(_input_params->sublist(name));
    }
    else
    {
        return Teuchos::null;
    }
}

//---------------------------------------------------------------------------//
void ParameterDatabase::extractSublists()
{
    // Check if we are using new input format. This will be removed once
    // the transition to the new format is complete.
    if (_input_params->isParameter("Use New Input Format"))
    {
        _use_new_input = _input_params->get<bool>("Use New Input Format");
    }

    // Create sublists from new format
    if (_use_new_input)
    {
        extractSublistsNew();
    }

    // Get sublists directly in old format.
    else
    {
        extractSublistsOld();
    }

    // Store the MPI communicator in the "User Data" ParameterList.
    _user_params->set<Teuchos::RCP<const Teuchos::Comm<int>>>("Comm", _comm);
}

//---------------------------------------------------------------------------//
void ParameterDatabase::extractSublistsOld()
{
    // Get required sublists.
    _mesh_params = requiredSublist("Mesh");
    _physics_params = requiredSublist("Physics Blocks");
    _block_mapping_params = requiredSublist("Block ID to Physics ID Mapping");
    _bc_params = requiredSublist("Boundary Conditions");
    _ic_params = requiredSublist("Initial Conditions");
    _closure_params = requiredSublist("Closure Models");
    _user_params = requiredSublist("User Data");
    _transient_solver_params = requiredSublist("Tempus");
    _linear_solver_params = requiredSublist("Linear Solver");

    // Get optional sublists.
    _profiling_params = optionalSublist("Profiling");
    _scalar_params = optionalSublist("Scalar Parameters");
    _general_scalar_params = optionalSublist("General Scalar Parameters");
    _output_params = optionalSublist("Solution Output");
    _response_output_params = optionalSublist("Scalar Response Output");
    if (Teuchos::is_null(_response_output_params))
    {
        // Check old name for backward compatibility
        _response_output_params = optionalSublist("Volume Integral Output");
    }
    _read_restart_params = optionalSublist("Read Restart");
    _write_restart_params = optionalSublist("Write Restart");
    _write_matrix_params = optionalSublist("Write Matrix");
}

//---------------------------------------------------------------------------//
void ParameterDatabase::extractSublistsNew()
{
    // Get required sublists.
    _mesh_params = requiredSublist("Mesh");
    _assembly_params = requiredSublist("Equation Assembly");
    _bc_params = requiredSublist("Boundary Conditions");
    _ic_params = requiredSublist("Initial Conditions");
    _closure_params = requiredSublist("Closure Models");
    _user_params = requiredSublist("User Data");
    _transient_solver_params = requiredSublist("Tempus");
    _linear_solver_params = requiredSublist("Linear Solver");

    // Get optional sublists.
    _profiling_params = optionalSublist("Profiling");
    _scalar_params = optionalSublist("Scalar Parameters");
    _general_scalar_params = optionalSublist("General Scalar Parameters");
    _output_params = optionalSublist("Solution Output");
    _response_output_params = optionalSublist("Scalar Response Output");
    if (Teuchos::is_null(_response_output_params))
    {
        // Check old name for backward compatibility
        _response_output_params = optionalSublist("Volume Integral Output");
    }
    _read_restart_params = optionalSublist("Read Restart");
    _write_restart_params = optionalSublist("Write Restart");
    _write_matrix_params = optionalSublist("Write Matrix");

    // Make auxiliary parameter lists.
    _physics_params
        = Teuchos::parameterList(_input_params->sublist("Physics Blocks"));
    _block_mapping_params = Teuchos::parameterList(
        _input_params->sublist("Block ID to Physics ID Mapping"));
}

//---------------------------------------------------------------------------//
Teuchos::RCP<const Teuchos::MpiComm<int>> ParameterDatabase::comm() const
{
    return _comm;
}

//---------------------------------------------------------------------------//
// Main list accessor.
Teuchos::RCP<Teuchos::ParameterList> ParameterDatabase::allParameters() const
{
    return _input_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList> ParameterDatabase::meshParameters() const
{
    return _mesh_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::assemblyParameters() const
{
    return _assembly_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::physicsParameters() const
{
    return _physics_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList> ParameterDatabase::scalarParameters() const
{
    return _scalar_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::generalScalarParameters() const
{
    return _general_scalar_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::blockMappingParameters() const
{
    return _block_mapping_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::boundaryConditionParameters() const
{
    return _bc_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::initialConditionParameters() const
{
    return _ic_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::closureModelParameters() const
{
    return _closure_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::responseOutputParameters() const
{
    return _response_output_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList> ParameterDatabase::userParameters() const
{
    return _user_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList> ParameterDatabase::outputParameters() const
{
    return _output_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::readRestartParameters() const
{
    return _read_restart_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::writeRestartParameters() const
{
    return _write_restart_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::writeMatrixParameters() const
{
    return _write_matrix_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::profilingParameters() const
{
    return _profiling_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::transientSolverParameters() const
{
    return _transient_solver_params;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
ParameterDatabase::linearSolverParameters() const
{
    return _linear_solver_params;
}

//---------------------------------------------------------------------------//
bool ParameterDatabase::useNewInputFormat() const
{
    return _use_new_input;
}

//---------------------------------------------------------------------------//

} // end namespace Parameter
} // end namespace VertexCFD

//---------------------------------------------------------------------------//

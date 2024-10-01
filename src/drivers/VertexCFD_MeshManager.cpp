#include "VertexCFD_MeshManager.hpp"
#include "mesh/VertexCFD_Mesh_StkReaderFactory.hpp"

#include <mpi.h>

namespace VertexCFD
{
//---------------------------------------------------------------------------//
MeshManager::MeshManager(const Parameter::ParameterDatabase& parameter_db,
                         const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm)
    : _comm(comm)
{
    auto mesh_params = parameter_db.meshParameters();

    if ("File" == mesh_params->get<std::string>("Mesh Input Type"))
    {
        _mesh_factory = Teuchos::rcp(new VertexCFD::Mesh::StkReaderFactory());
        auto file_params
            = Teuchos::parameterList(mesh_params->sublist("File"));
        _mesh_factory->setParameterList(file_params);
    }
    else if ("Inline" == mesh_params->get<std::string>("Mesh Input Type"))
    {
        auto inline_params
            = Teuchos::parameterList(mesh_params->sublist("Inline"));

        // Initialize integer to store mesh dimension
        int mesh_dimension = 0;

        if ("Tri3" == inline_params->get<std::string>("Element Type"))
        {
            _mesh_factory
                = Teuchos::rcp(new panzer_stk::SquareTriMeshFactory());
            mesh_dimension = 2;
        }
        else if ("Quad4" == inline_params->get<std::string>("Element Type"))
        {
            _mesh_factory
                = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory());
            mesh_dimension = 2;
        }
        else if ("Tet4" == inline_params->get<std::string>("Element Type"))
        {
            _mesh_factory = Teuchos::rcp(new panzer_stk::CubeTetMeshFactory());
            mesh_dimension = 3;
        }
        else if ("Hex8" == inline_params->get<std::string>("Element Type"))
        {
            _mesh_factory = Teuchos::rcp(new panzer_stk::CubeHexMeshFactory());
            mesh_dimension = 3;
        }
        else
        {
            throw std::runtime_error(
                "Invalid inline element type. Valid options are "
                "'Tri3', "
                "'Quad4', 'Tet4' and 'Hex8'");
        }
        auto inline_mesh_params
            = Teuchos::parameterList(inline_params->sublist("Mesh"));

        if (mesh_dimension > 1)
        {
            inline_mesh_params->set<int>("X Procs", -1);
            inline_mesh_params->set<int>("Y Procs", -1);
        }
        if (mesh_dimension > 2)
        {
            inline_mesh_params->set<int>("Z Procs", -1);
        }
        _mesh_factory->setParameterList(inline_mesh_params);
    }
    else
    {
        throw std::runtime_error(
            "Invalid mesh input type. Valid options are 'File' and 'Inline'");
    }

    _mesh = _mesh_factory->buildUncommitedMesh(Teuchos::getRawMpiComm(*_comm));
}

//---------------------------------------------------------------------------//
void MeshManager::completeMeshConstruction()
{
    _mesh_factory->completeMeshConstruction(*_mesh,
                                            Teuchos::getRawMpiComm(*_comm));
    _conn_manager = Teuchos::rcp(new panzer_stk::STKConnManager(_mesh));
}

//---------------------------------------------------------------------------//
Teuchos::RCP<const Teuchos::MpiComm<int>> MeshManager::comm() const
{
    return _comm;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer_stk::STK_Interface> MeshManager::mesh() const
{
    return _mesh;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer_stk::STKConnManager>
MeshManager::connectivityManager() const
{
    return _conn_manager;
}

//---------------------------------------------------------------------------//
int MeshManager::spaceDimension() const
{
    return static_cast<int>(_mesh->getDimension());
}

//---------------------------------------------------------------------------//

} // end namespace VertexCFD

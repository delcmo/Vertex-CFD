#ifndef VERTEXCFD_MESHMANAGER_HPP
#define VERTEXCFD_MESHMANAGER_HPP

#include "parameters/VertexCFD_ParameterDatabase.hpp"

#include <PanzerAdaptersSTK_config.hpp>
#include <Panzer_STKConnManager.hpp>
#include <Panzer_STK_CubeHexMeshFactory.hpp>
#include <Panzer_STK_CubeTetMeshFactory.hpp>
#include <Panzer_STK_IOClosureModel_Factory_TemplateBuilder.hpp>
#include <Panzer_STK_SetupLOWSFactory.hpp>
#include <Panzer_STK_SquareQuadMeshFactory.hpp>
#include <Panzer_STK_SquareTriMeshFactory.hpp>
#include <Panzer_STK_WorksetFactory.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
//---------------------------------------------------------------------------//
class MeshManager
{
  public:
    MeshManager(const Parameter::ParameterDatabase& parameter_db,
                const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm);

    void completeMeshConstruction();

    Teuchos::RCP<const Teuchos::MpiComm<int>> comm() const;
    Teuchos::RCP<panzer_stk::STK_Interface> mesh() const;
    Teuchos::RCP<panzer_stk::STKConnManager> connectivityManager() const;

    int spaceDimension() const;

  private:
    Teuchos::RCP<const Teuchos::MpiComm<int>> _comm;
    Teuchos::RCP<panzer_stk::STK_MeshFactory> _mesh_factory;
    Teuchos::RCP<panzer_stk::STK_Interface> _mesh;
    Teuchos::RCP<panzer_stk::STKConnManager> _conn_manager;
};

//---------------------------------------------------------------------------//

} // end namespace VertexCFD

#endif // end VERTEXCFD_MESHMANAGER_HPP

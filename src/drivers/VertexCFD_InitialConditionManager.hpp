#ifndef VERTEXCFD_INITIALCONDITIONMANAGER_HPP
#define VERTEXCFD_INITIALCONDITIONMANAGER_HPP

#include "VertexCFD_MeshManager.hpp"
#include "VertexCFD_PhysicsManager.hpp"

#include "initial_conditions/VertexCFD_InitialConditionFactory_TemplateBuilder.hpp"
#include "mesh/VertexCFD_Mesh_Restart.hpp"
#include "parameters/VertexCFD_ParameterDatabase.hpp"

#include <Panzer_InitialCondition_Builder.hpp>

#include <Thyra_VectorSpaceBase.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
//---------------------------------------------------------------------------//
class InitialConditionManager
{
  public:
    InitialConditionManager(
        const Teuchos::RCP<Parameter::ParameterDatabase>& parameter_db,
        const Teuchos::RCP<MeshManager>& mesh_manager);

    double initialTime() const;

    template<int NumSpaceDim>
    void
    applyInitialConditions(const std::integral_constant<int, NumSpaceDim>&,
                           const PhysicsManager& physics_manager,
                           Teuchos::RCP<Thyra::VectorBase<double>>& x,
                           Teuchos::RCP<Thyra::VectorBase<double>>& x_dot) const;

  private:
    Teuchos::RCP<Parameter::ParameterDatabase> _parameter_db;
    Teuchos::RCP<MeshManager> _mesh_manager;
    Teuchos::RCP<Mesh::RestartReader> _restart_reader;
    bool _do_restart;
    double _t_init;
};

//---------------------------------------------------------------------------//

} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITIONMANAGER_HPP

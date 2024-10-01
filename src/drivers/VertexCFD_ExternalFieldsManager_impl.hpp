#ifndef VERTEXCFD_EXTERNALFIELDSMANAGER_IMPL_HPP
#define VERTEXCFD_EXTERNALFIELDSMANAGER_IMPL_HPP

#include "VertexCFD_InitialConditionManager.hpp"
#include "VertexCFD_MeshManager.hpp"
#include "VertexCFD_PhysicsManager.hpp"
#include "parameters/VertexCFD_ParameterDatabase.hpp"

#include <Panzer_ReadOnlyVector_GlobalEvaluationData.hpp>

#include <Thyra_DefaultSpmdVector.hpp>

namespace VertexCFD
{
//---------------------------------------------------------------------------//
template<class Traits>
template<int NumSpaceDim>
ExternalFieldsManager<Traits>::ExternalFieldsManager(
    const std::integral_constant<int, NumSpaceDim>& num_space_dim,
    const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
    const std::string& filename)
{
    auto parameter_db
        = Teuchos::rcp(new Parameter::ParameterDatabase(comm, filename));
    auto mesh_manager = Teuchos::rcp(new MeshManager(*parameter_db, comm));
    auto physics_manager = Teuchos::rcp(
        new PhysicsManager(num_space_dim, parameter_db, mesh_manager));
    physics_manager->setupModel();

    _global_indexer = physics_manager->dofManager();

    auto ic_manager = Teuchos::rcp(
        new InitialConditionManager(parameter_db, mesh_manager));
    Teuchos::RCP<Thyra::VectorBase<double>> solution;
    Teuchos::RCP<Thyra::VectorBase<double>> solution_dot;
    ic_manager->applyInitialConditions(
        num_space_dim, *physics_manager, solution, solution_dot);

    // Create a global evaluation container for the field data.
    Teuchos::RCP<panzer::ReadOnlyVector_GlobalEvaluationData> ged;
    ged = physics_manager->linearObjectFactory()->buildReadOnlyDomainContainer();
    ged->setOwnedVector(solution);

    // Gather the ghosted vector of field data.
    ged->globalToGhost(0);

    // Get the local vector data.
    auto ghosted_vector
        = Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(
            ged->getGhostedVector());
    auto ghosted_data_host = ghosted_vector->getLocalSubVector();

    // Thyra only provides the ghosted data via a host-side array.
    // We need to copy this data to a device view so that it can be accessed
    // from the device in kernel below.
    _ghosted_field_data = Kokkos::View<double*, PHX::Device>(
        "ghosted_field_data", ghosted_data_host.subDim());
    auto ghost_mirror = Kokkos::create_mirror(_ghosted_field_data);
    for (int i = 0; i < ghosted_data_host.subDim(); ++i)
        ghost_mirror(i) = ghosted_data_host[i];
    Kokkos::deep_copy(_ghosted_field_data, ghost_mirror);
}

//---------------------------------------------------------------------------//
template<class Traits>
Teuchos::RCP<const panzer::GlobalIndexer>
ExternalFieldsManager<Traits>::globalIndexer() const
{
    return _global_indexer;
}

//---------------------------------------------------------------------------//
template<class Traits>
Kokkos::View<double*, PHX::Device>
ExternalFieldsManager<Traits>::ghostedFieldData() const
{
    return _ghosted_field_data;
}

//---------------------------------------------------------------------------//

} // end namespace VertexCFD

#endif // end VERTEXCFD_EXTERNALFIELDSMANAGER_IMPL_HPP

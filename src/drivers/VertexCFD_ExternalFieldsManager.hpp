#ifndef VERTEXCFD_EXTERNALFIELDSMANAGER_HPP
#define VERTEXCFD_EXTERNALFIELDSMANAGER_HPP

#include <Thyra_VectorBase.hpp>

#include <Panzer_GlobalIndexer.hpp>

#include <Phalanx_KokkosDeviceTypes.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_RCP.hpp>

#include <Kokkos_Core.hpp>

#include <string>
#include <type_traits>

namespace VertexCFD
{
//---------------------------------------------------------------------------//
template<class Traits>
class ExternalFieldsManager
{
  public:
    template<int NumSpaceDim>
    ExternalFieldsManager(
        const std::integral_constant<int, NumSpaceDim>& num_space_dim,
        const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
        const std::string& filename);

    Teuchos::RCP<const panzer::GlobalIndexer> globalIndexer() const;
    Kokkos::View<double*, PHX::Device> ghostedFieldData() const;

  private:
    // External global indexer (DOF manager).
    Teuchos::RCP<const panzer::GlobalIndexer> _global_indexer;

    // Gathered local field values.
    Kokkos::View<double*, PHX::Device> _ghosted_field_data;
};

//---------------------------------------------------------------------------//

} // end namespace VertexCFD

#include "VertexCFD_ExternalFieldsManager_impl.hpp"

#endif // end VERTEXCFD_EXTERNALFIELDSMANAGER_HPP

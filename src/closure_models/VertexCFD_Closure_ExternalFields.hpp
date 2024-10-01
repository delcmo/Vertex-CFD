#ifndef VERTEXCFD_CLOSURE_EXTERNALFIELDS_HPP
#define VERTEXCFD_CLOSURE_EXTERNALFIELDS_HPP

#include <drivers/VertexCFD_ExternalFieldsManager.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_PureBasis.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_RCP.hpp>

#include <Kokkos_Core.hpp>

#include <string>
#include <vector>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Add external fields from another physics manager as a closure model. This
// will gather the fields and put them at the basis points.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ExternalFields : public panzer::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ExternalFields(const std::string& evaluator_name,
                   const Teuchos::RCP<const ExternalFieldsManager<Traits>>&
                       external_fields_manager,
                   const std::vector<std::string>& external_field_names,
                   const Teuchos::RCP<const panzer::PureBasis>& basis);

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& fm);

    void evaluateFields(typename Traits::EvalData d);

  private:
    int _num_field;

  public:
    std::vector<PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS>>
        _external_fields;

  private:
    Teuchos::RCP<const panzer::GlobalIndexer> _global_indexer;
    std::vector<int> _field_ids;
    Kokkos::View<double*, PHX::Device> _ghosted_field_data;
    Kokkos::View<int**, PHX::Device> _scratch_lids;
    std::vector<Kokkos::View<int*, PHX::Device>> _scratch_offsets;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_EXTERNALFIELDS_HPP

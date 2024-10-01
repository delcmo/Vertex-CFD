#ifndef VERTEXCFD_CLOSURE_METRICTENSOR_HPP
#define VERTEXCFD_CLOSURE_METRICTENSOR_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class MetricTensor : public panzer::EvaluatorWithBaseImpl<Traits>,
                     public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim, panzer::Dim>
        _metric_tensor;

    MetricTensor(const panzer::IntegrationRule& ir);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    template<int NumSpaceDim>
    KOKKOS_INLINE_FUNCTION void
    operator()(std::integral_constant<int, NumSpaceDim>,
               const int cell,
               const int point) const;

  private:
    const int _ir_degree;
    const int _num_topo_dim;
    int _ir_index;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim, panzer::Dim>
        _jacobian;

    Kokkos::View<double**, PHX::mem_space> _element_map;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_METRICTENSOR_HPP

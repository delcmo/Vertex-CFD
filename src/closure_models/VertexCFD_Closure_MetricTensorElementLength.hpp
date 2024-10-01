#ifndef VERTEXCFD_CLOSURE_METRICTENSORELEMENTLENGTH_HPP
#define VERTEXCFD_CLOSURE_METRICTENSORELEMENTLENGTH_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class MetricTensorElementLength
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> _element_length;

    MetricTensorElementLength(const panzer::IntegrationRule& ir,
                              const std::string& prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim, panzer::Dim>
        _metric_tensor;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_METRICTENSORELEMENTLENGTH_HPP

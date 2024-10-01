#ifndef VERTEXCFD_CLOSURE_ELEMENTLENGTH_HPP
#define VERTEXCFD_CLOSURE_ELEMENTLENGTH_HPP

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
// Element length (dimensional) evaluator.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ElementLength : public panzer::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ElementLength(const panzer::IntegrationRule& ir,
                  const std::string& prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> _element_length;

  private:
    PHX::MDField<double, panzer::Cell, panzer::BASIS, panzer::IP, panzer::Dim>
        _grad_basis;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_ELEMENTLENGTH_HPP

#ifndef VERTEXCFD_CLOSURE_MEASUREELEMENTLENGTH_HPP
#define VERTEXCFD_CLOSURE_MEASUREELEMENTLENGTH_HPP

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
class MeasureElementLength : public panzer::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> _element_length;

    MeasureElementLength(const panzer::IntegrationRule& ir,
                         const std::string& prefix = "");

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    int _ir_degree;
    int _ir_index;

    PHX::MDField<const double, panzer::Cell, panzer::Point> _cell_det;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_MEASUREELEMENTLENGTH_HPP

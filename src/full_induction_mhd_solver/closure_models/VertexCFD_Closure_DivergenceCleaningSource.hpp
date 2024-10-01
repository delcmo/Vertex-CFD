#ifndef VERTEXCFD_CLOSURE_DIVERGENCECLEANINGSOURCE_HPP
#define VERTEXCFD_CLOSURE_DIVERGENCECLEANINGSOURCE_HPP

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
// Hyperbolic divergence cleaning source term for the magnetic correction
// potential equation
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class DivergenceCleaningSource : public panzer::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    DivergenceCleaningSource(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _div_cleaning_potential_source;

  private:
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_scalar_magnetic_potential;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_DIVERGENCECLEANINGSOURCE_HPP

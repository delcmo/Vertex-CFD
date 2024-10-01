#ifndef VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALINSULATINGWALL_HPP
#define VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALINSULATINGWALL_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ElectricPotentialInsulatingWall
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ElectricPotentialInsulatingWall(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _boundary_electric_potential;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_electric_potential;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _electric_potential;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_electric_potential;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALINSULATINGWALL_HPP

#ifndef VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALFIXED_HPP
#define VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALFIXED_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ElectricPotentialFixed : public panzer::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ElectricPotentialFixed(const panzer::IntegrationRule& ir,
                           const Teuchos::ParameterList& bc_params);

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
    int _num_grad_dim;
    double _time;
    double _time_init;
    double _time_final;
    double _a_sc;
    double _b_sc;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_electric_potential;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALFIXED_HPP

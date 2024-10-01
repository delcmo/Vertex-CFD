#ifndef VERTEXCFD_CLOSURE_INDUCTIONRESISTIVEFLUX_HPP
#define VERTEXCFD_CLOSURE_INDUCTIONRESISTIVEFLUX_HPP

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Multi-dimension induction resistive flux evaluation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class InductionResistiveFlux : public panzer::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    InductionResistiveFlux(
        const panzer::IntegrationRule& ir,
        const MHDProperties::FullInductionMHDProperties& mhd_props,
        const std::string& flux_prefix = "",
        const std::string& gradient_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _induction_flux;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _magnetic_correction_potential_flux;

  private:
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>, 3>
        _total_magnetic_field;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _resistivity;

    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        3>
        _grad_total_magnetic_field;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_resistivity;

    bool _variable_resistivity;
    bool _solve_magn_corr;
    double _magnetic_permeability;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INDUCTIONRESISTIVEFLUX_HPP

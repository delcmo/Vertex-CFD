#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGASOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGASOURCE_HPP

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
// Source term for the Wilcox (2006) K-Omega turbulence model with/without
// limiter
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleKOmegaSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleKOmegaSource(const panzer::IntegrationRule& ir,
                               const Teuchos::ParameterList& user_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_kinetic_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_specific_dissipation_rate;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    double _beta_star;
    double _gamma;
    double _beta_0;
    double _sigma_d;
    bool _limit_production;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_prod;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_dest;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_prod;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_dest;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_cross;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGASOURCE_HPP

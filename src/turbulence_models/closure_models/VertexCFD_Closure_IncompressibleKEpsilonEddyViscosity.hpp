#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONEDDYVISCOSITY_HPP

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
// Turbulent eddy viscosity for standard K-Epsilon turbulence model
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleKEpsilonEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleKEpsilonEddyViscosity(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_dissipation_rate;

    double _C_nu;
    int _num_grad_dim;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONEDDYVISCOSITY_HPP

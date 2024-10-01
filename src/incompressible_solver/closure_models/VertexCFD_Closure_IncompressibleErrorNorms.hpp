#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEERRORNORMS_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEERRORNORMS_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Compute error norms between exact solution and numerical solutions
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleErrorNorms : public panzer::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleErrorNorms(const panzer::IntegrationRule& ir,
                             const Teuchos::ParameterList& user_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _L1_error_continuity;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _L1_error_momentum;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _L1_error_energy;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _L2_error_continuity;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _L2_error_momentum;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _L2_error_energy;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _volume;

  private:
    PHX::MDField<const double, panzer::Cell, panzer::Point>
        _exact_lagrange_pressure;
    Kokkos::Array<PHX::MDField<const double, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _exact_velocity;
    PHX::MDField<const double, panzer::Cell, panzer::Point> _exact_temperature;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _temperature;

    bool _use_temp;
};

//---------------------------------------------------------------------------//

} // namespace ClosureModel
} // end namespace VertexCFD

#include "VertexCFD_Closure_IncompressibleErrorNorms_impl.hpp"

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEERRORNORMS_HPP

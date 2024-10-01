#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLENOSLIP_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLENOSLIP_HPP

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

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
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleNoSlip : public panzer::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleNoSlip(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& bc_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _boundary_lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_temperature;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _boundary_velocity;

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _boundary_grad_velocity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_temperature;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;

    bool _solve_temp;
    double _T_wall;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLENOSLIP_HPP

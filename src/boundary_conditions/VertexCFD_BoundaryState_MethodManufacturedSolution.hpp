#ifndef VERTEXCFD_BOUNDARYSTATE_METHODMANUFACTUREDSOLUTION_HPP
#define VERTEXCFD_BOUNDARYSTATE_METHODMANUFACTUREDSOLUTION_HPP

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
// Strong boundary condition for MMS
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class MethodManufacturedSolution
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;
    static constexpr int num_coeff = 2 * (num_space_dim + 1);

    MethodManufacturedSolution(const panzer::IntegrationRule& ir);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _boundary_lagrange_pressure;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _boundary_velocity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_temperature;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_lagrange_pressure;
    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _boundary_grad_velocity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_temperature;

  private:
    int _ir_degree;
    int _ir_index;

    Kokkos::Array<double, num_coeff> _phi_coeff;
    Kokkos::Array<Kokkos::Array<double, num_coeff>, num_space_dim> _vel_coeff;
    Kokkos::Array<double, num_coeff> _T_coeff;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#include "VertexCFD_BoundaryState_MethodManufacturedSolution_impl.hpp"

#endif // VERTEXCFD_BOUNDARYSTATE_METHODMANUFACTUREDSOLUTION_HPP

#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEINLETOUTLET_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEINLETOUTLET_HPP

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
template<class EvalType, class Traits, int NumSpaceDim>
class TurbulenceInletOutlet : public panzer::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    TurbulenceInletOutlet(const panzer::IntegrationRule& ir,
                          const Teuchos::ParameterList& bc_params,
                          const std::string variable_name);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_variable;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_variable;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _variable;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    double _inlet_value;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_TURBULENCEINLETOUTLET_HPP

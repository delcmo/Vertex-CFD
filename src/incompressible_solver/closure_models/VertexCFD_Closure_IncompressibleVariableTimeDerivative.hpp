#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEVARIABLETIMEDERIVATIVE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEVARIABLETIMEDERIVATIVE_HPP

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
// Time derivative term for incompressible turbulence variable equation
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleVariableTimeDerivative
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleVariableTimeDerivative(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params);

    void evaluateFields(typename Traits::EvalData d) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    std::string _variable_name;
    std::string _equation_name;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_var_eq;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_var;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEVARIABLETIMEDERIVATIVE_HPP

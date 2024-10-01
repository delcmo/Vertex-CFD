#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_HPP

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
// Turbulent eddy viscosity for WALE LES turbulence model
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleWALEEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleWALEEddyViscosity(const panzer::IntegrationRule& ir,
                                    const Teuchos::ParameterList& user_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;

    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    double _C_k;
    double _C_w;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_sgs;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_HPP

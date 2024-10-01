#ifndef VERTEXCFD_CLOSURE_FULLINDUCTIONMODELERRORNORMS_HPP
#define VERTEXCFD_CLOSURE_FULLINDUCTIONMODELERRORNORMS_HPP

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
class FullInductionModelErrorNorms
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    FullInductionModelErrorNorms(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _L1_error_induced;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _L2_error_induced;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _volume;

  private:
    Kokkos::Array<PHX::MDField<const double, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _exact_induced_magnetic_field;

    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _induced_magnetic_field;
};

//---------------------------------------------------------------------------//

} // namespace ClosureModel
} // end namespace VertexCFD

#include "VertexCFD_Closure_FullInductionModelErrorNorms_impl.hpp"

#endif // VERTEXCFD_CLOSURE_FULLINDUCTIONMODELERRORNORMS_HPP

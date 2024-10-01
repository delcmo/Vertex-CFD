#ifndef VERTEXCFD_CLOSURE_VECTORFIELDDIVERGENCE_HPP
#define VERTEXCFD_CLOSURE_VECTORFIELDDIVERGENCE_HPP

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
// Closure to compute divergence of a vector field
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class VectorFieldDivergence : public panzer::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    VectorFieldDivergence(const panzer::IntegrationRule& ir,
                          const std::string& field_name,
                          const std::string& closure_name);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    int _num_grad_dim;
    bool _use_abs;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_vector_field;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _vector_field_divergence;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_VECTORFIELDDIVERGENCE_HPP

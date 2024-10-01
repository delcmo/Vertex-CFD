#ifndef VERTEXCFD_UTILS_SCALARTOVECTOR_HPP
#define VERTEXCFD_UTILS_SCALARTOVECTOR_HPP

#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>

namespace VertexCFD
{
namespace Utils
{
//---------------------------------------------------------------------------//
template<typename EvalType, typename DimTag>
class ScalarToVector : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                       public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
  private:
    using ScalarT = typename EvalType::ScalarT;

  public:
    ScalarToVector(const panzer::IntegrationRule& ir,
                   const std::string& field_name,
                   const int num_scalars,
                   const bool time_deriv);

    void evaluateFields(typename panzer::Traits::EvalData);

  private:
    // Dependent scalar fields
    std::vector<PHX::MDField<const ScalarT, panzer::Cell, panzer::Point>>
        _scalar_fields;
    std::vector<PHX::MDField<const ScalarT, panzer::Cell, panzer::Point>>
        _scalar_dxdt_fields;
    std::vector<
        PHX::MDField<const ScalarT, panzer::Cell, panzer::Point, panzer::Dim>>
        _scalar_grad_fields;

  public:
    // Evaluated vector fields
    PHX::MDField<ScalarT, panzer::Cell, panzer::Point, DimTag> _vector_fields;
    PHX::MDField<ScalarT, panzer::Cell, panzer::Point, DimTag> _vector_dxdt_fields;
    PHX::MDField<ScalarT, panzer::Cell, panzer::Point, panzer::Dim, DimTag>
        _vector_grad_fields;
};

//---------------------------------------------------------------------------//

} // namespace Utils
} // namespace VertexCFD

#include "VertexCFD_Utils_ScalarToVector_impl.hpp"

#endif // VERTEXCFD_UTILS_SCALARTOVECTOR_HPP

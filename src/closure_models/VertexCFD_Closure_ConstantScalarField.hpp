#ifndef VERTEXCFD_CLOSURE_CONSTANTSCALARFIELD_HPP
#define VERTEXCFD_CLOSURE_CONSTANTSCALARFIELD_HPP

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
// Closure model to assign a constant value to a named scalar field
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ConstantScalarField : public panzer::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ConstantScalarField(const panzer::IntegrationRule& ir,
                        const std::string& field_name,
                        const double field_value);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _scalar_field;

  private:
    double _field_value;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONSTANTSCALARFIELD_HPP

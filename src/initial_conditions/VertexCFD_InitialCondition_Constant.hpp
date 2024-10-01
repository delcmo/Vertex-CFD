#ifndef VERTEXCFD_INITIALCONDITION_CONSTANT_HPP
#define VERTEXCFD_INITIALCONDITION_CONSTANT_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_PureBasis.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class Constant : public panzer::EvaluatorWithBaseImpl<Traits>,
                 public PHX::EvaluatorDerived<EvalType, Traits>

{
  public:
    using scalar_type = typename EvalType::ScalarT;

    Constant(const Teuchos::ParameterList& params,
             const panzer::PureBasis& basis);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS> _ic;

  private:
    scalar_type _value;
};

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_CONSTANT_HPP

#ifndef VERTEXCFD_SCALARPARAMETEREVALUATOR_HPP
#define VERTEXCFD_SCALARPARAMETEREVALUATOR_HPP

#include "VertexCFD_ScalarParameterManager.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalData.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_FieldTag.hpp>

#include <Teuchos_RCP.hpp>

#include <string>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ScalarParameterEvaluator : public panzer::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ScalarParameterEvaluator(
        const Teuchos::RCP<ScalarParameterManager<EvalType>>& param_manager,
        const Teuchos::RCP<panzer::GlobalData>& global_data);

    void evaluateFields(typename Traits::EvalData workset) override;

  private:
    Teuchos::RCP<PHX::FieldTag> _param_update_trigger;
    Teuchos::RCP<ScalarParameterManager<EvalType>> _param_manager;
    Teuchos::RCP<panzer::GlobalData> _global_data;
};

//---------------------------------------------------------------------------//

} // end namespace Parameter
} // end namespace VertexCFD

#endif // end VERTEXCFD_SCALARPARAMETEREVALUATOR_HPP

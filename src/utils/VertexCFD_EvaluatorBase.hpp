#ifndef VERTEXCFD_EVALUATORBASE_HPP
#define VERTEXCFD_EVALUATORBASE_HPP

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_FieldTag.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class EvaluatorBase : public panzer::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    EvaluatorBase();

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm) override;
    void preEvaluate(typename Traits::PreEvalData d) override;
    void evaluateFields(typename Traits::EvalData d) override;
    void postEvaluate(typename Traits::PostEvalData d) override;

  protected:
    virtual void postRegistrationSetupImpl(typename Traits::SetupData d,
                                           PHX::FieldManager<Traits>& vm);
    virtual void preEvaluateImpl(typename Traits::PreEvalData d);
    virtual void evaluateFieldsImpl(typename Traits::EvalData d) = 0;
    virtual void postEvaluateImpl(typename Traits::PostEvalData d);

  private:
    Teuchos::RCP<PHX::FieldTag> _param_update_trigger;
};

//---------------------------------------------------------------------------//

} // end namespace VertexCFD

#include "VertexCFD_EvaluatorBase_impl.hpp"

#endif // end VERTEXCFD_EVALUATORBASE_HPP

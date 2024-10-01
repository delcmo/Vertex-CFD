#ifndef VERTEXCFD_EVALUATORBASE_IMPL_HPP
#define VERTEXCFD_EVALUATORBASE_IMPL_HPP

#include <Panzer_Dimension.hpp>

#include <Phalanx_DataLayout_MDALayout.hpp>
#include <Phalanx_FieldTag_Tag.hpp>

namespace VertexCFD
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
EvaluatorBase<EvalType, Traits>::EvaluatorBase()
{
    // Add scalar parameter trigger
    auto dummy_layout = Teuchos::rcp(new PHX::MDALayout<panzer::Dummy>(0));
    _param_update_trigger = Teuchos::rcp(
        new PHX::Tag<scalar_type>("scalar_parameter_eval", dummy_layout));
    this->addDependentField(*_param_update_trigger);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void EvaluatorBase<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData d, PHX::FieldManager<Traits>& vm)
{
    this->postRegistrationSetupImpl(d, vm);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void EvaluatorBase<EvalType, Traits>::preEvaluate(typename Traits::PreEvalData d)
{
    this->preEvaluateImpl(d);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void EvaluatorBase<EvalType, Traits>::evaluateFields(typename Traits::EvalData d)
{
    this->evaluateFieldsImpl(d);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void EvaluatorBase<EvalType, Traits>::postEvaluate(
    typename Traits::PostEvalData d)
{
    this->postEvaluateImpl(d);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void EvaluatorBase<EvalType, Traits>::postRegistrationSetupImpl(
    typename Traits::SetupData, PHX::FieldManager<Traits>&)
{
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void
    EvaluatorBase<EvalType, Traits>::preEvaluateImpl(typename Traits::PreEvalData)
{
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void EvaluatorBase<EvalType, Traits>::postEvaluateImpl(
    typename Traits::PostEvalData)
{
}

//---------------------------------------------------------------------------//

} // end namespace VertexCFD

#endif // end VERTEXCFD_EVALUATORBASE_IMPL_HPP

#ifndef VERTEXCFD_SCALARPARAMETEREVALUATOR_IMPL_HPP
#define VERTEXCFD_SCALARPARAMETEREVALUATOR_IMPL_HPP

#include <Panzer_Dimension.hpp>

#include <Phalanx_DataLayout_MDALayout.hpp>
#include <Phalanx_FieldTag_Tag.hpp>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ScalarParameterEvaluator<EvalType, Traits>::ScalarParameterEvaluator(
    const Teuchos::RCP<ScalarParameterManager<EvalType>>& param_manager,
    const Teuchos::RCP<panzer::GlobalData>& global_data)
    : _param_manager(param_manager)
    , _global_data(global_data)
{
    auto dummy_layout = Teuchos::rcp(new PHX::MDALayout<panzer::Dummy>(0));
    _param_update_trigger = Teuchos::rcp(
        new PHX::Tag<scalar_type>("scalar_parameter_eval", dummy_layout));
    this->addEvaluatedField(*_param_update_trigger);
    this->setName("Scalar Parameter Evaluation");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ScalarParameterEvaluator<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _param_manager->update(*_global_data, workset);
}

//---------------------------------------------------------------------------//

} // end namespace Parameter
} // end namespace VertexCFD

#endif // end VERTEXCFD_SCALARPARAMETEREVALUATOR_IMPL_HPP

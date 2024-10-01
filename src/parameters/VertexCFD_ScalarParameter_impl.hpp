#ifndef VERTEXCFD_SCALARPARAMETER_IMPL_HPP
#define VERTEXCFD_SCALARPARAMETER_IMPL_HPP

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
template<class EvalType>
ScalarParameter<EvalType>::ScalarParameter(const std::string& name,
                                           scalar_type& ref_to_parameter)
    : _name(name)
    , _ref_to_parameter(ref_to_parameter)
{
}

//---------------------------------------------------------------------------//
template<class EvalType>
const std::string& ScalarParameter<EvalType>::name() const
{
    return _name;
}

//---------------------------------------------------------------------------//
template<class EvalType>
void ScalarParameter<EvalType>::update(const panzer::GlobalData& global_data)
{
    _ref_to_parameter = global_data.pl->getValue<EvalType>(_name);
}

//---------------------------------------------------------------------------//

} // namespace Parameter
} // namespace VertexCFD

#endif // end VERTEXCFD_SCALARPARAMETER_IMPL_HPP

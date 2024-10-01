#ifndef VERTEXCFD_SCALARPARAMETEROBSERVER_IMPL_HPP
#define VERTEXCFD_SCALARPARAMETEROBSERVER_IMPL_HPP

#include "VertexCFD_GeneralScalarParameterInput.hpp"
#include "VertexCFD_ScalarParameterInput.hpp"

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
template<class EvalType>
void ScalarParameterObserver<EvalType>::registerParameter(
    const std::string& name,
    const double default_value,
    const Teuchos::ParameterList& plist,
    scalar_type& ref_to_parameter)
{
    // Sensitivity Parameter
    // General Parameter

    // Sensitivity scalar parameter. Initially assign the default
    // value. Parameterized values will get assigned during update(),
    // including those nominal values set in the input file. Sensitivity
    // scalar parameters have:
    //
    //     nominal value: provided by user, applied when model is not being
    //     parameterized w.r.t. given parameter
    //
    //     default value: provided by developer, applied when the user
    //     doesn't specify the parameter in any way
    if (plist.isType<ScalarParameterInput>(name))
    {
        auto param_input = plist.get<ScalarParameterInput>(name);
        _scalar_parameters.emplace_back(param_input.parameter_name,
                                        ref_to_parameter);
        ref_to_parameter = default_value;
    }

    // General scalar parameter. A default value is provided to use for all
    // blocks that aren't given their own parameter value. Block parameter
    // values will get assigned during update(), including those default
    // values set in the input file. General scalar parameters have:
    //
    //     block value: provided by user, applied on the given block
    //
    //     default value; provided by user, applied when no block value is
    //     given for a block
    //
    //     default value: provided by developer, applied when the user
    //     doesn't specify the parameter in any way
    else if (plist.isType<GeneralScalarParameterInput>(name))
    {
        auto param_input = plist.get<GeneralScalarParameterInput>(name);
        _general_parameters.emplace_back(param_input.parameter_name,
                                         ref_to_parameter);
        ref_to_parameter = default_value;
    }

    // Check for local parameter.
    else if (plist.isType<double>(name))
    {
        ref_to_parameter = plist.get<double>(name);
    }

    // Otherwise treat as local parameter and set default.
    else
    {
        ref_to_parameter = default_value;
    }
}

//---------------------------------------------------------------------------//
template<class EvalType>
void ScalarParameterObserver<EvalType>::update(
    const panzer::GlobalData& global_data,
    const panzer::Workset& workset,
    const std::unordered_map<std::string, std::unordered_map<std::string, double>>&
        general_parameter_data)
{
    for (auto& p : _scalar_parameters)
    {
        p.update(global_data);
    }
    for (auto& p : _general_parameters)
    {
        p.update(workset, general_parameter_data);
    }
    this->updateStateWithNewParameters();
}

//---------------------------------------------------------------------------//

} // namespace Parameter
} // namespace VertexCFD

#endif // end VERTEXCFD_SCALARPARAMETEROBSERVER_IMPL_HPP

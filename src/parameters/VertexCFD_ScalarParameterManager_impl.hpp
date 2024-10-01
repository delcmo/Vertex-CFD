#ifndef VERTEXCFD_SCALARPARAMETERMANAGER_IMPL_HPP
#define VERTEXCFD_SCALARPARAMETERMANAGER_IMPL_HPP

#include <utility>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
template<class EvalType>
ScalarParameterManager<EvalType>::ScalarParameterManager(
    const ParameterDatabase& parameter_db)
{
    // Extract the general_scalar parameters from the input.
    auto general_scalar_params = parameter_db.generalScalarParameters();
    if (Teuchos::nonnull(general_scalar_params))
    {
        // Loop through parameters
        for (const auto& gp : *general_scalar_params)
        {
            // Loop through block values
            std::unordered_map<std::string, double> block_values;
            auto param_values = general_scalar_params->sublist(gp.first);
            for (const auto& bv : param_values)
            {
                // Insert block name and value.
                block_values.emplace(bv.first, bv.second.getValue<double>(0));
            }

            // Insert parameter name and block values.
            _general_parameter_data.emplace(gp.first, std::move(block_values));
        }
    }
}

//---------------------------------------------------------------------------//
template<class EvalType>
void ScalarParameterManager<EvalType>::addObserver(
    const Teuchos::RCP<ScalarParameterObserver<EvalType>>& observer)
{
    _observers.emplace_back(observer);
}

//---------------------------------------------------------------------------//
template<class EvalType>
void ScalarParameterManager<EvalType>::update(
    const panzer::GlobalData& global_data, const panzer::Workset& workset)
{
    for (auto& o : _observers)
    {
        o->update(global_data, workset, _general_parameter_data);
    }
}

//---------------------------------------------------------------------------//

} // namespace Parameter
} // namespace VertexCFD

#endif // end VERTEXCFD_SCALARPARAMETERMANAGER_IMPL_HPP

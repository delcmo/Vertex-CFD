#ifndef VERTEXCFD_CLOSUREMODELFACTORY_HPP
#define VERTEXCFD_CLOSUREMODELFACTORY_HPP

#include <Panzer_ClosureModel_Factory.hpp>

#include <Teuchos_ParameterList.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
class Factory : public panzer::ClosureModelFactory<EvalType>
{
  public:
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
    buildClosureModels(const std::string& model_id,
                       const Teuchos::ParameterList& model_params,
                       const panzer::FieldLayoutLibrary& fl,
                       const Teuchos::RCP<panzer::IntegrationRule>& ir,
                       const Teuchos::ParameterList& default_params,
                       const Teuchos::ParameterList& user_params,
                       const Teuchos::RCP<panzer::GlobalData>& global_data,
                       PHX::FieldManager<panzer::Traits>& fm) const override;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSUREMODELFACTORY_HPP

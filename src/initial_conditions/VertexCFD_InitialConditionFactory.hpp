#ifndef VERTEXCFD_INITIALCONDITIONFACTORY_HPP
#define VERTEXCFD_INITIALCONDITIONFACTORY_HPP

#include <Panzer_ClosureModel_Factory.hpp>
#include <Panzer_STK_Interface.hpp>

#include <Teuchos_ParameterList.hpp>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
class Factory : public panzer::ClosureModelFactory<EvalType>
{
  public:
    Factory(Teuchos::RCP<const panzer_stk::STK_Interface> mesh);

    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
    buildClosureModels(const std::string& block_id,
                       const Teuchos::ParameterList& block_params,
                       const panzer::FieldLayoutLibrary& fl,
                       const Teuchos::RCP<panzer::IntegrationRule>& ir,
                       const Teuchos::ParameterList& default_params,
                       const Teuchos::ParameterList& user_data,
                       const Teuchos::RCP<panzer::GlobalData>& global_data,
                       PHX::FieldManager<panzer::Traits>& fm) const override;

  private:
    Teuchos::RCP<const panzer_stk::STK_Interface> _mesh;
};

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITIONFACTORY_HPP

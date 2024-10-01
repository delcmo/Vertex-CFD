#ifndef VERTEXCFD_BOUNDARYCONDITION_STORNGDIRICHLETMMS_HPP
#define VERTEXCFD_BOUNDARYCONDITION_STORNGDIRICHLETMMS_HPP

#include <Panzer_BCStrategy_Dirichlet_DefaultImpl.hpp>
#include <Panzer_PhysicsBlock.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_FieldManager.hpp>

#include <Teuchos_RCP.hpp>

#include <vector>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType>
class StrongDirichletMMS
    : public panzer::BCStrategy_Dirichlet_DefaultImpl<EvalType>
{
  public:
    StrongDirichletMMS(const panzer::BC& bc,
                       const Teuchos::RCP<panzer::GlobalData>& global_data);

    void setup(const panzer::PhysicsBlock& side_pb,
               const Teuchos::ParameterList& user_data) override;

    void buildAndRegisterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& pb,
        const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& factory,
        const Teuchos::ParameterList& models,
        const Teuchos::ParameterList& user_data) const override;

  private:
    std::vector<panzer::StrPureBasisPair> _dofs;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#include "VertexCFD_BCStrategy_StrongDirichletMMS_impl.hpp"

#endif // end VERTEXCFD_BOUNDARYCONDITION_STORNGDIRICHLETMMS_HPP

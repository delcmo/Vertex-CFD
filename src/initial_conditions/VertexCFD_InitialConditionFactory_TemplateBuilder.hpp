#ifndef VERTEXCFD_INITIALCONDITIONFACTORY_TEMPLATEBUILDER_HPP
#define VERTEXCFD_INITIALCONDITIONFACTORY_TEMPLATEBUILDER_HPP

#include "VertexCFD_InitialConditionFactory.hpp"

#include <Panzer_ClosureModel_Factory_Base.hpp>
#include <Panzer_STK_Interface.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class FactoryTemplateBuilder
{
  public:
    FactoryTemplateBuilder(Teuchos::RCP<const panzer_stk::STK_Interface> mesh)
        : _mesh{mesh}
    {
    }

    template<typename EvalT>
    Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
    {
        auto ic_factory = Teuchos::rcp(new Factory<EvalT, NumSpaceDim>(_mesh));
        return Teuchos::rcp_static_cast<panzer::ClosureModelFactoryBase>(
            ic_factory);
    }

  private:
    Teuchos::RCP<const panzer_stk::STK_Interface> _mesh;
};

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITIONFACTORY_TEMPLATEBUILDER_HPP

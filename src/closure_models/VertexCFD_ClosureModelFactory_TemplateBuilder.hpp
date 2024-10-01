#ifndef VERTEXCFD_CLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
#define VERTEXCFD_CLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP

#include "VertexCFD_ClosureModelFactory.hpp"

#include <Panzer_ClosureModel_Factory_Base.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class FactoryTemplateBuilder
{
  public:
    template<typename EvalT>
    Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
    {
        auto closure_factory = Teuchos::rcp(new Factory<EvalT, NumSpaceDim>{});
        return Teuchos::rcp_static_cast<panzer::ClosureModelFactoryBase>(
            closure_factory);
    }
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP

#ifndef VERTEXCFD_EQUATIONSET_FACTORY_HPP
#define VERTEXCFD_EQUATIONSET_FACTORY_HPP

#include "VertexCFD_EquationSet_Heat.hpp"
#include "VertexCFD_EquationSet_IncompressibleNavierStokes.hpp"

#include <Panzer_CellData.hpp>
#include <Panzer_EquationSet_Factory.hpp>
#include <Panzer_EquationSet_Factory_Defines.hpp>

namespace VertexCFD
{
namespace EquationSet
{
//---------------------------------------------------------------------------//

PANZER_DECLARE_EQSET_TEMPLATE_BUILDER(Heat, Heat)
PANZER_DECLARE_EQSET_TEMPLATE_BUILDER(IncompressibleNavierStokes,
                                      IncompressibleNavierStokes)

//---------------------------------------------------------------------------//
class Factory : public panzer::EquationSetFactory
{
  public:
    Teuchos::RCP<panzer::EquationSet_TemplateManager<panzer::Traits>>
    buildEquationSet(const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const int& default_integration_order,
                     const panzer::CellData& cell_data,
                     const Teuchos::RCP<panzer::GlobalData>& global_data,
                     const bool build_transient_support) const override
    {
        // This variable needs to have this exact name to work with the macro
        // called below.
        auto eq_set = Teuchos::rcp(
            new panzer::EquationSet_TemplateManager<panzer::Traits>);

        // The "found" variable is used in-place in the macro called below.
        bool found = false;

        // Call the macro for each equation set and check that we found it.
        PANZER_BUILD_EQSET_OBJECTS("Heat", Heat);
        PANZER_BUILD_EQSET_OBJECTS("IncompressibleNavierStokes",
                                   IncompressibleNavierStokes);
        if (!found)
        {
            throw std::runtime_error("Equation set not valid");
        }

        // Return the equation set
        return eq_set;
    }
};

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_FACTORY_HPP

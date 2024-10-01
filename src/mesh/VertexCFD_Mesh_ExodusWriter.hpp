#ifndef VERTEXCFD_MESH_EXODUSWRITER_HPP
#define VERTEXCFD_MESH_EXODUSWRITER_HPP

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_ResponseLibrary.hpp>
#include <Panzer_STK_Interface.hpp>
#include <Panzer_STK_ResponseEvaluatorFactory_SolutionWriter.hpp>
#include <Panzer_STK_Utilities.hpp>

#include <Thyra_VectorBase.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace Mesh
{
//---------------------------------------------------------------------------//
class ExodusWriter
{
  public:
    ExodusWriter(
        const Teuchos::RCP<panzer_stk::STK_Interface>& mesh,
        const Teuchos::RCP<const panzer::GlobalIndexer>& dof_manager,
        const Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits>>& lof,
        const Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits>>&
            response_library,
        const Teuchos::ParameterList& output_params);

    void
    writeSolution(const Teuchos::RCP<const Thyra::VectorBase<double>>& x,
                  const Teuchos::RCP<const Thyra::VectorBase<double>>& x_dot,
                  const double time = 0.0,
                  const double time_step = 0.0);

  private:
    enum class OutputType
    {
        Scalar,
        Vector
    };

    enum class OutputLocation
    {
        Node,
        Cell
    };

    void add_mesh_outputs(const Teuchos::ParameterList& params,
                          const OutputType output_type,
                          const OutputLocation output_location);

    Teuchos::RCP<panzer_stk::STK_Interface> _mesh;
    Teuchos::RCP<const panzer::GlobalIndexer> _dof_manager;
    Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits>> _lof;
    Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits>> _response_library;
};

//---------------------------------------------------------------------------//

} // end namespace Mesh
} // end namespace VertexCFD

#endif // end VERTEXCFD_MESH_EXODUSWRITER_HPP

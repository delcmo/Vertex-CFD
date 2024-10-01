#ifndef VERTEXCFD_SCALARPARAMETERMANAGER_HPP
#define VERTEXCFD_SCALARPARAMETERMANAGER_HPP

#include "VertexCFD_ParameterDatabase.hpp"
#include "VertexCFD_ScalarParameterObserver.hpp"

#include <Panzer_GlobalData.hpp>

#include <Teuchos_RCP.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
template<class EvalType>
class ScalarParameterManager
{
  public:
    // Construct a manager from the parameter database.
    ScalarParameterManager(const ParameterDatabase& parameter_db);

    // Assign a new observer to be managed by this manager.
    void
    addObserver(const Teuchos::RCP<ScalarParameterObserver<EvalType>>& observer);

    // Update the parameters in all observers owned by this manager with the
    // given global data. Also update the state of all observers with the new
    // parameter values.
    void update(const panzer::GlobalData& global_data,
                const panzer::Workset& workset);

  private:
    std::vector<Teuchos::RCP<ScalarParameterObserver<EvalType>>> _observers;
    std::unordered_map<std::string, std::unordered_map<std::string, double>>
        _general_parameter_data;
};

//---------------------------------------------------------------------------//

} // namespace Parameter
} // namespace VertexCFD

#endif // end VERTEXCFD_SCALARPARAMETERMANAGER_HPP

#ifndef VERTEXCFD_SCALARPARAMETEROBSERVER_HPP
#define VERTEXCFD_SCALARPARAMETEROBSERVER_HPP

#include "VertexCFD_GeneralScalarParameter.hpp"
#include "VertexCFD_ScalarParameter.hpp"

#include <Panzer_GlobalData.hpp>

#include <Teuchos_ParameterList.hpp>

#include <string>
#include <vector>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
template<class EvalType>
class ScalarParameterObserver
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    // Register a parameter with the observer. The parameter will be extracted
    // from the parameter list and assigned to the reference (note that this
    // reference is stored by the underlying parameter). If the parameter is
    // not in the list the provided default value will be used.
    void registerParameter(const std::string& name,
                           const double default_value,
                           const Teuchos::ParameterList& plist,
                           scalar_type& ref_to_parameter);

    // Update the value of all parameters owned by the observer with the
    // current state of the global data and the current workset.
    void
    update(const panzer::GlobalData& global_data,
           const panzer::Workset& workset,
           const std::unordered_map<std::string,
                                    std::unordered_map<std::string, double>>&
               general_parameter_data);

  protected:
    // After the parameters have been updated, update the state of the
    // observer as necessary to be consistent with the new parameter
    // values. This will always be called inside of update() and therefore the
    // parameter state is guaranteed to be up-to-date.
    virtual void updateStateWithNewParameters() = 0;

  private:
    std::vector<ScalarParameter<EvalType>> _scalar_parameters;
    std::vector<GeneralScalarParameter<EvalType>> _general_parameters;
};

//---------------------------------------------------------------------------//

} // namespace Parameter
} // namespace VertexCFD

#endif // end VERTEXCFD_SCALARPARAMETEROBSERVER_HPP

#ifndef VERTEXCFD_GENERALSCALARPARAMETER_HPP
#define VERTEXCFD_GENERALSCALARPARAMETER_HPP

#include "VertexCFD_GeneralScalarParameterInput.hpp"

#include <Panzer_Workset.hpp>

#include <Teuchos_ParameterList.hpp>

#include <string>
#include <unordered_map>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
template<class EvalType>
class GeneralScalarParameter
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    GeneralScalarParameter(const std::string& name,
                           scalar_type& ref_to_parameter);

    const std::string& name() const;

    void
    update(const panzer::Workset& workset,
           const std::unordered_map<std::string,
                                    std::unordered_map<std::string, double>>&
               general_scalar_params);

  private:
    std::string _name;
    scalar_type& _ref_to_parameter;
};

//---------------------------------------------------------------------------//

} // end namespace Parameter
} // end namespace VertexCFD

#endif // end VERTEXCFD_GENERALSCALARPARAMETER_HPP

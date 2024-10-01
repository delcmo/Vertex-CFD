#ifndef VERTEXCFD_SCALARPARAMETER_HPP
#define VERTEXCFD_SCALARPARAMETER_HPP

#include <Panzer_GlobalData.hpp>

#include <string>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
// Global scalar parameter. Parameters managed by this class will be
// updated from the parameter library.
//---------------------------------------------------------------------------//
template<class EvalType>
class ScalarParameter
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ScalarParameter(const std::string& name, scalar_type& ref_to_parameter);

    const std::string& name() const;

    void update(const panzer::GlobalData& global_data);

  private:
    std::string _name;
    scalar_type& _ref_to_parameter;
};

//---------------------------------------------------------------------------//

} // namespace Parameter
} // namespace VertexCFD

#endif // end VERTEXCFD_SCALARPARAMETER_HPP

#ifndef VERTEXCFD_SCALARPARAMETERINPUT_HPP
#define VERTEXCFD_SCALARPARAMETERINPUT_HPP

#include <Teuchos_TypeNameTraits.hpp>

#include <iostream>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
struct ScalarParameterInput
{
    std::string parameter_name;

    bool operator==(const ScalarParameterInput& rhs) const;
};

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& out, const ScalarParameterInput& input);

std::istream& operator>>(std::istream& in, ScalarParameterInput& input);

//---------------------------------------------------------------------------//

} // end namespace Parameter
} // end namespace VertexCFD

//---------------------------------------------------------------------------//
namespace Teuchos
{
template<>
class TypeNameTraits<VertexCFD::Parameter::ScalarParameterInput>
{
  public:
    static std::string name() { return "ScalarParameter"; }
};

} // end namespace Teuchos

//---------------------------------------------------------------------------//

#endif // end VERTEXCFD_SCALARPARAMETERINPUT_HPP

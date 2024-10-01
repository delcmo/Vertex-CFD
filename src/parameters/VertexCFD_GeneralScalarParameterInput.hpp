#ifndef VERTEXCFD_GENERALSCALARPARAMTERINPUT_HPP
#define VERTEXCFD_GENERALSCALARPARAMTERINPUT_HPP

#include <Teuchos_TypeNameTraits.hpp>

#include <iostream>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
struct GeneralScalarParameterInput
{
    std::string parameter_name;

    bool operator==(const GeneralScalarParameterInput& rhs) const;
};

//---------------------------------------------------------------------------//
std::ostream&
operator<<(std::ostream& out, const GeneralScalarParameterInput& input);

std::istream& operator>>(std::istream& in, GeneralScalarParameterInput& input);

//---------------------------------------------------------------------------//

} // end namespace Parameter
} // end namespace VertexCFD

//---------------------------------------------------------------------------//
namespace Teuchos
{
template<>
class TypeNameTraits<VertexCFD::Parameter::GeneralScalarParameterInput>
{
  public:
    static std::string name() { return "GeneralScalarParameter"; }
};

} // end namespace Teuchos

//---------------------------------------------------------------------------//

#endif // end VERTEXCFD_GENERALSCALARPARAMTERINPUT_HPP

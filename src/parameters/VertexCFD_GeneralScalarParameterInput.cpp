#include "VertexCFD_GeneralScalarParameterInput.hpp"

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
bool GeneralScalarParameterInput::operator==(
    const GeneralScalarParameterInput& rhs) const
{
    return parameter_name == rhs.parameter_name;
}

//---------------------------------------------------------------------------//
std::ostream&
operator<<(std::ostream& out, const GeneralScalarParameterInput& input)
{
    out << input.parameter_name;
    return out;
}

//---------------------------------------------------------------------------//
std::istream& operator>>(std::istream& in, GeneralScalarParameterInput& input)
{
    std::getline(in, input.parameter_name);
    return in;
}

//---------------------------------------------------------------------------//

} // end namespace Parameter
} // end namespace VertexCFD

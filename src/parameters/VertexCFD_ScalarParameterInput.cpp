#include "VertexCFD_ScalarParameterInput.hpp"

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
bool ScalarParameterInput::operator==(const ScalarParameterInput& rhs) const
{
    return parameter_name == rhs.parameter_name;
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& out, const ScalarParameterInput& input)
{
    out << input.parameter_name;
    return out;
}

//---------------------------------------------------------------------------//
std::istream& operator>>(std::istream& in, ScalarParameterInput& input)
{
    std::getline(in, input.parameter_name);
    return in;
}

//---------------------------------------------------------------------------//

} // end namespace Parameter
} // end namespace VertexCFD

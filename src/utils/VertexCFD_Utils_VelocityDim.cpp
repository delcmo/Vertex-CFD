#include "VertexCFD_Utils_VelocityDim.hpp"

namespace PHX
{

// Shortened version of tag when, e.g., printing DAGs
template<>
std::string print<VertexCFD::VelocityDim>()
{
    return "Vel";
}

} // namespace PHX

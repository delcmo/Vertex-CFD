#ifndef VERTEXCFD_UTILS_VELOCITYDIM_HPP
#define VERTEXCFD_UTILS_VELOCITYDIM_HPP

#include <Phalanx_ExtentTraits.hpp>

namespace VertexCFD
{

// PHX tag denoting velocity dimension for an MDField
struct VelocityDim
{
};

} // namespace VertexCFD

namespace PHX
{

// Shortened version of tag for, e.g., printing DAG
template<>
std::string print<VertexCFD::VelocityDim>();

} // namespace PHX

// Register tag as PHX extent
PHX_IS_EXTENT(VertexCFD::VelocityDim)

#endif // VERTEXCFD_UTILS_VELOCITYDIM_HPP

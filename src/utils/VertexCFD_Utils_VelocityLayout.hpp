#ifndef VERTEXCFD_UTILS_VELOCITYLAYOUT_HPP
#define VERTEXCFD_UTILS_VELOCITYLAYOUT_HPP

#include <Phalanx_DataLayout.hpp>

namespace VertexCFD
{
namespace Utils
{
// Build phalanx DataLayout corresponding to vector velocity field
Teuchos::RCP<PHX::DataLayout>
buildVelocityLayout(const Teuchos::RCP<const PHX::DataLayout>& scalar_layout,
                    int num_vel_dims);

// Build phalanx DataLayout corresponding to vector velocity gradient field
Teuchos::RCP<PHX::DataLayout>
buildVelocityGradLayout(const Teuchos::RCP<const PHX::DataLayout>& vector_layout,
                        int num_vel_dims);
} // namespace Utils
} // namespace VertexCFD

#endif // VERTEXCFD_UTILS_VELOCITYLAYOUT_HPP

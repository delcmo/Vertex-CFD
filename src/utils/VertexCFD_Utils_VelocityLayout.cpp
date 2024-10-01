#include "VertexCFD_Utils_VelocityLayout.hpp"
#include "VertexCFD_Utils_VelocityDim.hpp"

#include <Panzer_Dimension.hpp>

#include <Phalanx_DataLayout_MDALayout.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace Utils
{
Teuchos::RCP<PHX::DataLayout>
buildVelocityLayout(const Teuchos::RCP<const PHX::DataLayout>& scalar_layout,
                    int num_vel_dims)
{
    return Teuchos::rcp(
        new PHX::MDALayout<panzer::Cell, panzer::Point, VelocityDim>(
            scalar_layout->extent(0), scalar_layout->extent(1), num_vel_dims));
}

Teuchos::RCP<PHX::DataLayout> buildVelocityGradLayout(
    const Teuchos::RCP<const PHX::DataLayout>& vector_layout, int num_vel_dims)
{
    return Teuchos::rcp(
        new PHX::MDALayout<panzer::Cell, panzer::Point, panzer::Dim, VelocityDim>(
            vector_layout->extent(0),
            vector_layout->extent(1),
            vector_layout->extent(2),
            num_vel_dims));
}

} // namespace Utils
} // namespace VertexCFD

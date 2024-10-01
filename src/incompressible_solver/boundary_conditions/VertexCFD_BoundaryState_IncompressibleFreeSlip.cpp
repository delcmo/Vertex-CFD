#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BoundaryState_IncompressibleFreeSlip.hpp"
#include "VertexCFD_BoundaryState_IncompressibleFreeSlip_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::IncompressibleFreeSlip)

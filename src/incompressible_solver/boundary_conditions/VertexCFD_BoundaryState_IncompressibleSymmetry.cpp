#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BoundaryState_IncompressibleSymmetry.hpp"
#include "VertexCFD_BoundaryState_IncompressibleSymmetry_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::IncompressibleSymmetry)

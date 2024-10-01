#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BoundaryState_TurbulenceInletOutlet.hpp"
#include "VertexCFD_BoundaryState_TurbulenceInletOutlet_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::TurbulenceInletOutlet)

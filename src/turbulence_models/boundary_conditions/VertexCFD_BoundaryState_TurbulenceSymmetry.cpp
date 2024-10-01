#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BoundaryState_TurbulenceSymmetry.hpp"
#include "VertexCFD_BoundaryState_TurbulenceSymmetry_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::BoundaryCondition::TurbulenceSymmetry)

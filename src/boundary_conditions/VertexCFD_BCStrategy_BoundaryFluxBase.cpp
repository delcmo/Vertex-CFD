#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BCStrategy_BoundaryFluxBase.hpp"
#include "VertexCFD_BCStrategy_BoundaryFluxBase_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::BoundaryFluxBase)

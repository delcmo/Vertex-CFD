#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_InitialConditionFactory.hpp"
#include "VertexCFD_InitialConditionFactory_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::InitialCondition::Factory)

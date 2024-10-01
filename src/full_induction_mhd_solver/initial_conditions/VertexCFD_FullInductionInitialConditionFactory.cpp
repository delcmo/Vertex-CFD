#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_FullInductionInitialConditionFactory.hpp"
#include "VertexCFD_FullInductionInitialConditionFactory_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::InitialCondition::FullInductionICFactory)

#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_ClosureModelFactory.hpp"
#include "VertexCFD_ClosureModelFactory_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::ClosureModel::Factory)

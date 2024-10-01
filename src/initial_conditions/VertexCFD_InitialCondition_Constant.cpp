#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_InitialCondition_Constant.hpp"
#include "VertexCFD_InitialCondition_Constant_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::InitialCondition::Constant)
#include "VertexCFD_InitialCondition_Constant_impl.hpp"

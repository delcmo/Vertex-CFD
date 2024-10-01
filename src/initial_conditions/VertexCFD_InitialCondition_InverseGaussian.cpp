#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_InitialCondition_InverseGaussian.hpp"
#include "VertexCFD_InitialCondition_InverseGaussian_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::InitialCondition::InverseGaussian)

#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "closure_models/VertexCFD_Closure_VectorFieldDivergence.hpp"
#include "closure_models/VertexCFD_Closure_VectorFieldDivergence_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::ClosureModel::VectorFieldDivergence)

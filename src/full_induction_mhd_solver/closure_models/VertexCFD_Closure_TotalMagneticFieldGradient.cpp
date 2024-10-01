#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_TotalMagneticFieldGradient.hpp"
#include "VertexCFD_Closure_TotalMagneticFieldGradient_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::ClosureModel::TotalMagneticFieldGradient)

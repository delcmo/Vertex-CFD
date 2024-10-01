#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_MagneticCorrectionDampingSource.hpp"
#include "VertexCFD_Closure_MagneticCorrectionDampingSource_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::ClosureModel::MagneticCorrectionDampingSource)

#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleVariableTimeDerivative.hpp"
#include "VertexCFD_Closure_IncompressibleVariableTimeDerivative_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::ClosureModel::IncompressibleVariableTimeDerivative)

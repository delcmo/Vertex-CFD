#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleTaylorGreenVortexExactSolution.hpp"
#include "VertexCFD_Closure_IncompressibleTaylorGreenVortexExactSolution_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::ClosureModel::IncompressibleTaylorGreenVortexExactSolution)

#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleKEpsilonEddyViscosity.hpp"
#include "VertexCFD_Closure_IncompressibleKEpsilonEddyViscosity_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::ClosureModel::IncompressibleKEpsilonEddyViscosity)

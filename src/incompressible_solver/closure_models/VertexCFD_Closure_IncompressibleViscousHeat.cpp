#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleViscousHeat.hpp"
#include "VertexCFD_Closure_IncompressibleViscousHeat_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::ClosureModel::IncompressibleViscousHeat)

#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_ElectricPotentialBoundaryState_Factory.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::ElectricPotentialBoundaryStateFactory)

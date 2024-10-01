#ifndef VERTEXCFD_INCOMPRESSIBLEBOUNDARYSTATE_FACTORY_HPP
#define VERTEXCFD_INCOMPRESSIBLEBOUNDARYSTATE_FACTORY_HPP

#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleCavityLid.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleDirichlet.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleFreeSlip.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleLaminarFlow.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleNoSlip.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressiblePressureOutflow.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleRotatingWall.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleSymmetry.hpp"

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleBoundaryStateFactory
{
  public:
    static Teuchos::RCP<PHX::Evaluator<Traits>>
    create(const panzer::IntegrationRule& ir,
           const Teuchos::ParameterList& bc_params,
           const Teuchos::ParameterList& user_params)
    {
        // Space dimension
        constexpr int num_space_dim = NumSpaceDim;

        // Equation of state
        Teuchos::ParameterList fluid_prop_list
            = user_params.sublist("Fluid Properties");
        const bool build_temp_equ
            = user_params.isType<bool>("Build Temperature Equation")
                  ? user_params.get<bool>("Build Temperature Equation")
                  : false;
        const bool build_buoyancy
            = user_params.isType<bool>("Build Buoyancy Source")
                  ? user_params.get<bool>("Build Buoyancy Source")
                  : false;
        const bool build_elec_pot_equ
            = user_params.isType<bool>("Build Electric Potential Equation")
                  ? user_params.get<bool>("Build Electric Potential Equation")
                  : false;

        fluid_prop_list.set<bool>("Build Temperature Equation", build_temp_equ);
        fluid_prop_list.set<bool>("Build Buoyancy Source", build_buoyancy);
        fluid_prop_list.set<bool>("Build Electric Potential Equation",
                                  build_elec_pot_equ);
        FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

        // Loop over boundary conditions
        Teuchos::RCP<PHX::Evaluator<Traits>> state;
        bool found_model = false;

        if (bc_params.isType<std::string>("Type"))
        {
            if (bc_params.get<std::string>("Type") == "No-Slip")
            {
                state = Teuchos::rcp(
                    new IncompressibleNoSlip<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Dirichlet")
            {
                state = Teuchos::rcp(
                    new IncompressibleDirichlet<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Pressure Outflow")
            {
                state = Teuchos::rcp(
                    new IncompressiblePressureOutflow<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Free Slip")
            {
                state = Teuchos::rcp(
                    new IncompressibleFreeSlip<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Symmetry")
            {
                state = Teuchos::rcp(
                    new IncompressibleSymmetry<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Rotating Wall")
            {
                state = Teuchos::rcp(
                    new IncompressibleRotatingWall<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Laminar Flow")
            {
                state = Teuchos::rcp(
                    new IncompressibleLaminarFlow<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Cavity Lid")
            {
                state = Teuchos::rcp(
                    new IncompressibleCavityLid<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params));
                found_model = true;
            }
        }

        if (!found_model)
        {
            std::string msg = "\n\nBoundary state "
                              + bc_params.get<std::string>("Type")
                              + " failed to build.\n";
            msg += "The boundary conditions implemented in VertexCFD are:\n";
            msg += "No-Slip,\n";
            msg += "Dirichlet,\n";
            msg += "Free Slip,\n";
            msg += "Pressure Outflow,\n";
            msg += "Rotating Wall,\n";
            msg += "Laminar Flow,\n";
            msg += "Symmetry,\n";
            msg += "Cavity Lid\n";
            msg += "\n";
            throw std::runtime_error(msg);
        }

        return state;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INCOMPRESSIBLEBOUNDARYSTATE_FACTORY_HPP

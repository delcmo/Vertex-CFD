#ifndef VERTEXCFD_INCOMPRESSIBLECLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_INCOMPRESSIBLECLOSUREMODELFACTORY_IMPL_HPP

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleBuoyancySource.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConstantSource.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConvectiveFlux.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleErrorNorms.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleLocalTimeStepSize.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressiblePlanarPoiseuilleExact.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleRotatingAnnulusExact.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleShearVariables.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleTaylorGreenVortexExactSolution.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleTimeDerivative.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleViscousFlux.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleViscousHeat.hpp"
#include "incompressible_solver/closure_models/VertexCFD_IncompressibleClosureModelFactory.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleFactory<EvalType, NumSpaceDim>::buildClosureModel(
    const std::string& closure_type,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& user_params,
    const Teuchos::ParameterList& /**closure_params**/,
    const bool use_turbulence_model,
    bool& found_model,
    std::string& error_msg,
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
        evaluators)
{
    // Define local variables
    constexpr int num_space_dim = NumSpaceDim;
    Teuchos::RCP<PHX::Evaluator<panzer::Traits>> eval;

    // Fluid properties
    Teuchos::ParameterList fluid_prop_list
        = user_params.sublist("Fluid Properties");
    const bool build_temp_equ
        = user_params.isType<bool>("Build Temperature Equation")
              ? user_params.get<bool>("Build Temperature Equation")
              : false;
    const bool build_buoyancy_source
        = user_params.isType<bool>("Build Buoyancy Source")
              ? user_params.get<bool>("Build Buoyancy Source")
              : false;
    const bool build_elec_pot_equ
        = user_params.isType<bool>("Build Electric Potential Equation")
              ? user_params.get<bool>("Build Electric Potential Equation")
              : false;
    fluid_prop_list.set<bool>("Build Temperature Equation", build_temp_equ);
    fluid_prop_list.set<bool>("Build Buoyancy Source", build_buoyancy_source);
    fluid_prop_list.set<bool>("Build Electric Potential Equation",
                              build_elec_pot_equ);
    FluidProperties::ConstantFluidProperties incompressible_fluidprop_params
        = FluidProperties::ConstantFluidProperties(fluid_prop_list);

    // Closure models
    if (closure_type == "IncompressibleTimeDerivative")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleTimeDerivative<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleConvectiveFlux")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleConvectiveFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleViscousFlux")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleViscousFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir,
                incompressible_fluidprop_params,
                user_params,
                use_turbulence_model));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleLocalTimeStepSize")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleLocalTimeStepSize<EvalType,
                                                panzer::Traits,
                                                num_space_dim>(*ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleConstantSource")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleConstantSource<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params, user_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleBuoyancySource")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleBuoyancySource<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params, user_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleViscousHeat")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleViscousHeat<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleRotatingAnnulusExact")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleRotatingAnnulusExact<EvalType,
                                                   panzer::Traits,
                                                   num_space_dim>(
                *ir, incompressible_fluidprop_params, user_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressiblePlanarPoiseuilleExact")
    {
        auto eval = Teuchos::rcp(
            new IncompressiblePlanarPoiseuilleExact<EvalType,
                                                    panzer::Traits,
                                                    num_space_dim>(
                *ir, incompressible_fluidprop_params, user_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleErrorNorm")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleErrorNorms<EvalType, panzer::Traits, num_space_dim>(
                *ir, user_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleTaylorGreenVortexExactSolution")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleTaylorGreenVortexExactSolution<EvalType,
                                                             panzer::Traits,
                                                             num_space_dim>(
                *ir, incompressible_fluidprop_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleShearVariables")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleShearVariables<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    // Initialize 'error_msg' with list of closure models for incompressible
    // NS equations
    error_msg += "IncompressibleBuoyancySource\n";
    error_msg += "IncompressibleConstantSource\n";
    error_msg += "IncompressibleConvectiveFlux\n";
    error_msg += "IncompressibleErrorNorm\n";
    error_msg += "IncompressibleLocalTimeStepSize\n";
    error_msg += "IncompressiblePlanarPoiseuilleExact\n";
    error_msg += "IncompressibleRotatingAnnulusExact\n";
    error_msg += "IncompressibleShearVariables\n";
    error_msg += "IncompressibleTimeDerivative\n";
    error_msg += "IncompressibleViscousFlux\n";
    error_msg += "IncompressibleViscousHeat\n";
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_INCOMPRESSIBLECLOSUREMODELFACTORY_IMPL_HPP

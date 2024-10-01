#ifndef VERTEXCFD_CLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_CLOSUREMODELFACTORY_IMPL_HPP

#include "VertexCFD_Closure_ElementLength.hpp"
#include "VertexCFD_Closure_MeasureElementLength.hpp"
#include "VertexCFD_Closure_MethodManufacturedSolution.hpp"
#include "VertexCFD_Closure_MethodManufacturedSolutionSource.hpp"
#include "VertexCFD_Closure_MetricTensor.hpp"
#include "VertexCFD_Closure_MetricTensorElementLength.hpp"
#include "VertexCFD_Closure_SingularValueElementLength.hpp"
#include "VertexCFD_Closure_VectorFieldDivergence.hpp"
#include "VertexCFD_Closure_WallDistance.hpp"

#include "incompressible_solver/closure_models/VertexCFD_IncompressibleClosureModelFactory.hpp"

#include "turbulence_models/closure_models/VertexCFD_TurbulenceClosureModelFactory.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_FullInductionClosureModelFactory.hpp"

#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricCurrentDensity.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricPotentialCrossProductFlux.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricPotentialDiffusionFlux.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ExternalMagneticField.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_HartmannProblemExact.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_LorentzForce.hpp"

#include "utils/VertexCFD_Utils_VectorizeOutputFieldNames.hpp"

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
Factory<EvalType, NumSpaceDim>::buildClosureModels(
    const std::string& model_id,
    const Teuchos::ParameterList& model_params,
    const panzer::FieldLayoutLibrary&,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList&,
    const Teuchos::ParameterList& user_params,
    const Teuchos::RCP<panzer::GlobalData>&,
    PHX::FieldManager<panzer::Traits>&) const
{
    auto evaluators = Teuchos::rcp(
        new std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>);

    constexpr int num_space_dim = NumSpaceDim;

    if (!model_params.isSublist(model_id))
    {
        throw std::runtime_error("Closure model id not in list");
    }

    // Turbulence model parameters
    TurbulenceFactory<EvalType, NumSpaceDim> tm_factory;
    bool use_turbulence_model = false;
    const std::string turbulence_model_name
        = user_params.isType<std::string>("Turbulence Model")
              ? user_params.get<std::string>("Turbulence Model")
              : "No Turbulence Model";

    if (turbulence_model_name != "No Turbulence Model")
    {
        tm_factory.buildClosureModel(
            ir, user_params, turbulence_model_name, evaluators);
        use_turbulence_model = true;
    }

    // Incompressible equation of state
    // TODO: the following logic remains until factory model functions
    // are added for the MHD equations and the turbulence models.
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
    FluidProperties::ConstantFluidProperties incompressible_fluidprop_params(
        fluid_prop_list);

    // Incompressible factory model objects
    IncompressibleFactory<EvalType, NumSpaceDim> incomp_factory;
    std::string incomp_error_msg = "None";

    // Full induction solver factory objects
    FullInductionFactory<EvalType, NumSpaceDim> full_induction_factory;
    std::string full_ind_error_msg = "None";

    // Closure model block in XML input file
    const Teuchos::ParameterList& closure_model_list
        = model_params.sublist(model_id);
    for (const auto& closure_model : closure_model_list)
    {
        bool found_model = false;

        const auto closure_name = closure_model.first;
        const auto& closure_params
            = Teuchos::getValue<Teuchos::ParameterList>(closure_model.second);

        if (closure_params.isType<std::string>("Type"))
        {
            const auto closure_type = closure_params.get<std::string>("Type");

            // Incompressible factory
            incomp_factory.buildClosureModel(closure_type,
                                             ir,
                                             user_params,
                                             closure_params,
                                             use_turbulence_model,
                                             found_model,
                                             incomp_error_msg,
                                             evaluators);

            // Full induction MHD factory
            if (user_params.isSublist("Full Induction MHD Properties"))
            {
                full_induction_factory.buildClosureModel(closure_type,
                                                         ir,
                                                         user_params,
                                                         closure_params,
                                                         found_model,
                                                         full_ind_error_msg,
                                                         evaluators);
            }

            // MHD closure models
            if (closure_type == "ElectricPotentialDiffusionFlux")
            {
                auto eval = Teuchos::rcp(
                    new ElectricPotentialDiffusionFlux<EvalType, panzer::Traits>(
                        *ir, incompressible_fluidprop_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "ElectricPotentialCrossProductFlux")
            {
                auto eval = Teuchos::rcp(
                    new ElectricPotentialCrossProductFlux<EvalType,
                                                          panzer::Traits,
                                                          num_space_dim>(
                        *ir, incompressible_fluidprop_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "ElectricCurrentDensity")
            {
                auto eval
                    = Teuchos::rcp(new ElectricCurrentDensity<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>(
                        *ir, incompressible_fluidprop_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "LorentzForce")
            {
                auto eval = Teuchos::rcp(
                    new LorentzForce<EvalType, panzer::Traits, num_space_dim>(
                        *ir, incompressible_fluidprop_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "HartmannProblemExact")
            {
                auto eval = Teuchos::rcp(
                    new HartmannProblemExact<EvalType, panzer::Traits, num_space_dim>(
                        *ir, incompressible_fluidprop_params, user_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "ExternalMagneticField")
            {
                auto eval = Teuchos::rcp(
                    new ExternalMagneticField<EvalType, panzer::Traits>(
                        *ir, user_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "MetricTensor")
            {
                auto eval = Teuchos::rcp(
                    new MetricTensor<EvalType, panzer::Traits>(*ir));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "ElementLength")
            {
                auto eval = Teuchos::rcp(
                    new ElementLength<EvalType, panzer::Traits>(*ir));
                evaluators->push_back(eval);
                found_model = true;
            }
            else if (closure_type == "MetricTensorElementLength")
            {
                auto eval = Teuchos::rcp(
                    new MetricTensorElementLength<EvalType, panzer::Traits>(
                        *ir));
                evaluators->push_back(eval);
                found_model = true;
            }
            else if (closure_type == "MeasureElementLength")
            {
                auto eval = Teuchos::rcp(
                    new MeasureElementLength<EvalType, panzer::Traits>(*ir));
                evaluators->push_back(eval);
                found_model = true;
            }
            else if (closure_type == "SingularValueElementLength")
            {
                const auto method
                    = closure_params.get<std::string>("Element Length Method");
                auto eval = Teuchos::rcp(
                    new SingularValueElementLength<EvalType, panzer::Traits>(
                        *ir, method));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "MethodManufacturedSolution")
            {
                auto eval = Teuchos::rcp(
                    new MethodManufacturedSolution<EvalType,
                                                   panzer::Traits,
                                                   num_space_dim>(*ir));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "MethodManufacturedSolutionSource")
            {
                bool build_viscous_flux = false;
                if (user_params.isType<bool>("Build Viscous Flux"))
                {
                    build_viscous_flux
                        = user_params.get<bool>("Build Viscous Flux");
                }
                auto eval = Teuchos::rcp(
                    new MethodManufacturedSolutionSource<EvalType,
                                                         panzer::Traits,
                                                         num_space_dim>(
                        *ir,
                        build_viscous_flux,
                        incompressible_fluidprop_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "WallDistance")
            {
                auto eval = Teuchos::rcp(
                    new WallDistance<EvalType, panzer::Traits, num_space_dim>(
                        *ir,
                        user_params.get<Teuchos::RCP<MeshManager>>("MeshManage"
                                                                   "r"),
                        user_params.sublist("Wall List")));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (std::string::npos != closure_type.find("VectorFieldDivergence"))
            {
                const auto field_names
                    = closure_params.get<std::string>("Field Names");
                std::vector<std::string> tokens;
                panzer::StringTokenizer(tokens, field_names, ",", true);
                for (auto& field : tokens)
                {
                    auto eval = Teuchos::rcp(
                        new VectorFieldDivergence<EvalType,
                                                  panzer::Traits,
                                                  num_space_dim>(
                            *ir, field, closure_type));
                    evaluators->push_back(eval);
                }
                found_model = true;
            }
        }

        if (!found_model)
        {
            std::string msg = "Closure model " + closure_name
                              + " failed to build.\n";
            msg += "The closure models implemented in VertexCFD are:\n";
            msg += "ElectricCurrentDensity\n";
            msg += "ElectricPotentialCrossProductFlux\n";
            msg += "ElectricPotentialDiffusionFlux\n";
            msg += "ElementLength\n";
            msg += "ExternalMagneticField\n";
            msg += "LorentzForce\n";
            msg += "MeasureElementLength\n";
            msg += "MethodManufacturedSolution\n";
            msg += "MethodManufacturedSolutionSource\n";
            msg += "MetricTensor\n";
            msg += "MetricTensorElementLength\n";
            msg += "SingularValueElementLength\n";
            msg += "ThermalConductivity\n";
            msg += "VectorFieldDivergence\n";
            msg += "AbsVectorFieldDivergence\n";
            msg += "=================================\n";
            msg += "Incompressible closure models:\n";
            msg += incomp_error_msg;
            msg += "=================================\n";
            msg += "Full induction MHD closure models:\n";
            msg += full_ind_error_msg;

            throw std::runtime_error(msg);
        }
    }

    return evaluators;
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSUREMODELFACTORY_IMPL_HPP

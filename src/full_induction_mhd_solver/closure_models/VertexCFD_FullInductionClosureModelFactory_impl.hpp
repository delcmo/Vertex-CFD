#ifndef VERTEXCFD_FULLINDUCTIONCLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_FULLINDUCTIONCLOSUREMODELFACTORY_IMPL_HPP

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleVariableTimeDerivative.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_DivergenceCleaningSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_FullInductionLocalTimeStepSize.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_FullInductionModelErrorNorms.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_GodunovPowellSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConvectiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionResistiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MHDVortexProblemExact.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MagneticCorrectionDampingSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MagneticPressure.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticField.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticFieldGradient.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void FullInductionFactory<EvalType, NumSpaceDim>::buildClosureModel(
    const std::string& closure_type,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& user_params,
    const Teuchos::ParameterList& /*closure_params*/,
    bool& found_model,
    std::string& error_msg,
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
        evaluators)
{
    // Fluid properties
    Teuchos::ParameterList fluid_prop_list
        = user_params.sublist("Fluid Properties");
    const bool build_temp_equ
        = user_params.isType<bool>("Build Temperature Equation")
              ? user_params.get<bool>("Build Temperature Equation")
              : false;
    fluid_prop_list.set("Build Temperature Equation", build_temp_equ);
    FluidProperties::ConstantFluidProperties incompressible_fluidprop_params
        = FluidProperties::ConstantFluidProperties(fluid_prop_list);

    // Properties used by full induction MHD closure models
    const auto full_induction_params
        = user_params.sublist("Full Induction MHD Properties");
    MHDProperties::FullInductionMHDProperties mhd_props
        = MHDProperties::FullInductionMHDProperties(full_induction_params);
    const bool build_magn_corr = mhd_props.buildMagnCorr();

    // Closure models
    if (closure_type == "InductionConvectiveFlux")
    {
        auto eval = Teuchos::rcp(
            new InductionConvectiveFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir, mhd_props));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "InductionResistiveFlux")
    {
        auto eval = Teuchos::rcp(
            new InductionResistiveFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir, mhd_props));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "MagneticPressure")
    {
        auto eval = Teuchos::rcp(
            new MagneticPressure<EvalType, panzer::Traits>(*ir, mhd_props));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "FullInductionTimeDerivative")
    {
        std::vector<Teuchos::ParameterList> mhd_names_list_vct;
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            Teuchos::ParameterList ind_names_list;
            ind_names_list.set(
                "Field Name", "induced_magnetic_field_" + std::to_string(dim));
            ind_names_list.set("Equation Name",
                               "induction_" + std::to_string(dim));
            mhd_names_list_vct.push_back(ind_names_list);
        }
        if (build_magn_corr)
        {
            Teuchos::ParameterList magn_corr_names_list;
            magn_corr_names_list.set("Field Name", "scalar_magnetic_potential");
            magn_corr_names_list.set("Equation Name",
                                     "magnetic_correction_potential");
            mhd_names_list_vct.push_back(magn_corr_names_list);
        }
        for (auto& mhd_names_list : mhd_names_list_vct)
        {
            auto eval_time = Teuchos::rcp(
                new IncompressibleVariableTimeDerivative<EvalType, panzer::Traits>(
                    *ir, mhd_names_list));
            evaluators->push_back(eval_time);
        }
        found_model = true;
    }

    if (closure_type == "TotalMagneticField")
    {
        auto eval = Teuchos::rcp(
            new TotalMagneticField<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval);
        auto eval_grad = Teuchos::rcp(
            new TotalMagneticFieldGradient<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval_grad);
        found_model = true;
    }

    if (closure_type == "MHDVortexProblemExact")
    {
        auto eval = Teuchos::rcp(
            new MHDVortexProblemExact<EvalType, panzer::Traits, NumSpaceDim>(
                *ir, full_induction_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "FullInductionModelErrorNorm")
    {
        auto eval = Teuchos::rcp(
            new FullInductionModelErrorNorms<EvalType, panzer::Traits, NumSpaceDim>(
                *ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "DivergenceCleaningSource")
    {
        auto eval = Teuchos::rcp(
            new DivergenceCleaningSource<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "GodunovPowellSource")
    {
        auto eval = Teuchos::rcp(
            new GodunovPowellSource<EvalType, panzer::Traits, NumSpaceDim>(
                *ir, mhd_props));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "MagneticCorrectionDampingSource")
    {
        auto eval = Teuchos::rcp(
            new MagneticCorrectionDampingSource<EvalType, panzer::Traits>(
                *ir, mhd_props));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "FullInductionLocalTimeStepSize")
    {
        auto eval = Teuchos::rcp(
            new FullInductionLocalTimeStepSize<EvalType, panzer::Traits, NumSpaceDim>(
                *ir, incompressible_fluidprop_params, mhd_props));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "Resistivity")
    {
        if (mhd_props.variableResistivity())
        {
            throw std::runtime_error(
                "No closure models currently exist to evaluate variable "
                "resistivity. Use a constant resistivity only.");
        }
        else
        {
            auto eval = Teuchos::rcp(
                new ConstantScalarField<EvalType, panzer::Traits>(
                    *ir, "resistivity", mhd_props.resistivity()));
            evaluators->push_back(eval);
            found_model = true;
        }
    }

    // Initialize 'error_msg' with list of closure models for induction MHD
    // equations
    error_msg = "DivergenceCleaningSource\n";
    error_msg += "FullInductionLocalTimeStepSize\n";
    error_msg += "FullInductionModelErrorNorm\n";
    error_msg += "FullInductionTimeDerivative\n";
    error_msg += "GodunovPowellSource\n";
    error_msg += "InductionConvectiveFlux\n";
    error_msg += "InductionResistiveFlux\n";
    error_msg += "MagneticCorrectionDampingSource\n";
    error_msg += "MagneticPressure\n";
    error_msg += "MHDVortexProblemExact\n";
    error_msg += "Resistivity\n";
    error_msg += "TotalMagneticField\n";
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_FULLINDUCTIONCLOSUREMODELFACTORY_IMPL_HPP

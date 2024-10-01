#ifndef VERTEXCFD_EQUATIONSET_HEAT_IMPL_HPP
#define VERTEXCFD_EQUATIONSET_HEAT_IMPL_HPP

#include <Panzer_BasisIRLayout.hpp>
#include <Panzer_EvaluatorStyle.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_Integrator_BasisTimesScalar.hpp>
#include <Panzer_Integrator_GradBasisDotVector.hpp>

#include <stdexcept>
#include <vector>

namespace VertexCFD
{
namespace EquationSet
{
//---------------------------------------------------------------------------//
template<class EvalType>
Heat<EvalType>::Heat(const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const int& default_integration_order,
                     const panzer::CellData& cell_data,
                     const Teuchos::RCP<panzer::GlobalData>& global_data,
                     const bool build_transient_support)
    : panzer::EquationSet_DefaultImpl<EvalType>(params,
                                                default_integration_order,
                                                cell_data,
                                                global_data,
                                                build_transient_support)
    , _dof_name("temperature")
{
    // This equation set need not always be transient. Could solve Poisson
    if (!this->buildTransientSupport())
    {
        throw std::logic_error("Heat equation requires transient support");
    }

    // Set default parameter values and validate the inputs.
    Teuchos::ParameterList valid_parameters;
    this->setDefaultValidParameters(valid_parameters);
    valid_parameters.set(
        "Model ID", "", "Closure model id associated with this equation set");
    valid_parameters.set("Basis Order", 1, "Order of the basis");
    valid_parameters.set("Integration Order", 2, "Order of the integration");
    params->validateParametersAndSetDefaults(valid_parameters);

    // Extract parameters.
    const int basis_order = params->get<int>("Basis Order", 1);
    const int integration_order
        = params->get<int>("Integration Order", basis_order + 1);
    const auto model_id = params->get<std::string>("Model ID");

    // Get the number of space dimensions.
    const auto num_space_dim = cell_data.baseCellDimension();
    if (!(num_space_dim == 2 || num_space_dim == 3))
    {
        throw std::runtime_error("Number of space dimensions not supported");
    }

    this->addDOF(_dof_name, "HGrad", basis_order, integration_order);
    this->addDOFGrad(_dof_name);
    if (this->buildTransientSupport())
    {
        this->addDOFTimeDerivative(_dof_name);
    }
    this->addClosureModel(model_id);
    this->setupDOFs();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void Heat<EvalType>::buildAndRegisterEquationSetEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::FieldLibrary&,
    const Teuchos::ParameterList&) const
{
    const auto ir = this->getIntRuleForDOF(_dof_name);
    const auto basis = this->getBasisIRLayoutForDOF(_dof_name);

    std::vector<std::string> term_names;

    if (this->buildTransientSupport())
    {
        const std::string term_name{"RESIDUAL_" + _dof_name + "_TRANSIENT_OP"};
        term_names.push_back(term_name);
        const auto op{Teuchos::rcp(
            new panzer::Integrator_BasisTimesScalar<EvalType, panzer::Traits>(
                panzer::EvaluatorStyle::EVALUATES,
                term_name,
                "DXDT_" + _dof_name,
                *basis,
                *ir,
                1.0))};
        this->template registerEvaluator<EvalType>(fm, op);
    }

    {
        const std::string term_name{"RESIDUAL_" + _dof_name
                                    + "_CONDUCTIVE_TERM"};
        term_names.push_back(term_name);
        const std::vector<std::string> field_multipliers{
            "thermal_conductivity"};
        const auto op{Teuchos::rcp(
            new panzer::Integrator_GradBasisDotVector<EvalType, panzer::Traits>(
                panzer::EvaluatorStyle::EVALUATES,
                term_name,
                "GRAD_" + _dof_name,
                *basis,
                *ir,
                1.0,
                field_multipliers))};
        this->template registerEvaluator<EvalType>(fm, op);
    }

    {
        const auto term_name{"RESIDUAL_" + _dof_name + "_SOURCE_OP"};
        const auto op{Teuchos::rcp(
            new panzer::Integrator_BasisTimesScalar<EvalType, panzer::Traits>(
                panzer::EvaluatorStyle::EVALUATES,
                term_name,
                "SOURCE_" + _dof_name,
                *basis,
                *ir,
                -1.0))};
        this->template registerEvaluator<EvalType>(fm, op);
    }

    this->buildAndRegisterResidualSummationEvaluator(fm, _dof_name, term_names);
}

//---------------------------------------------------------------------------/
template<class EvalType>
std::string Heat<EvalType>::fieldName(const int dof) const
{
    if (dof > 0)
    {
        throw std::logic_error("Heat equation contributes a single DOF");
    }

    return _dof_name;
}

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_HEAT_IMPL_HPP

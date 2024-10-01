#ifndef VERTEXCFD_BOUNDARYCONDITION_BOUNDARYFLUXBASE_IMPL_HPP
#define VERTEXCFD_BOUNDARYCONDITION_BOUNDARYFLUXBASE_IMPL_HPP

#include "VertexCFD_BoundaryState_ViscousGradient.hpp"
#include "VertexCFD_BoundaryState_ViscousPenaltyParameter.hpp"
#include "VertexCFD_Integrator_BoundaryGradBasisDotVector.hpp"

#include <Panzer_DOF.hpp>
#include <Panzer_DOFGradient.hpp>
#include <Panzer_DotProduct.hpp>
#include <Panzer_Integrator_BasisTimesScalar.hpp>
#include <Panzer_Normals.hpp>
#include <Panzer_Sum.hpp>

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_DataLayout_MDALayout.hpp>
#include <Phalanx_MDField.hpp>

#include <map>
#include <string>
#include <vector>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
BoundaryFluxBase<EvalType, NumSpaceDim>::BoundaryFluxBase(
    const panzer::BC& bc, const Teuchos::RCP<panzer::GlobalData>& global_data)
    : panzer::BCStrategy<EvalType>(bc)
    , panzer::GlobalDataAcceptorDefaultImpl(global_data)
{
    // Initialize `bnd_prefix` to use with second-order flux
    bnd_prefix.insert({"BOUNDARY_", "BOUNDARY_"});
    bnd_prefix.insert({"PENALTY_BOUNDARY_", "PENALTY_"});
    bnd_prefix.insert({"SYMMETRY_BOUNDARY_", "SYMMETRY_"});
}

//---------------------------------------------------------------------------//
// Initialize class members
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::initialize(
    const panzer::PhysicsBlock& side_pb,
    std::unordered_map<std::string, std::string>& /**dof_eq_map**/)
{
    // Integration rule and integration order
    const auto& irules = side_pb.getIntegrationRules();
    if (irules.size() != 1)
    {
        throw std::runtime_error(
            "BCStrategy BoundaryFluxBis too many integration rules");
    }
    _integration_order = irules.begin()->second->order();
    _ir = Teuchos::rcp(
        new panzer::IntegrationRule(_integration_order, side_pb.cellData()));
}

//---------------------------------------------------------------------------//
// Get integration basis for a variable with name `dof_name`.
template<class EvalType, int NumSpaceDim>
auto BoundaryFluxBase<EvalType, NumSpaceDim>::getIntegrationBasis(
    const panzer::PhysicsBlock& side_pb, const std::string& dof_name) const
{
    const auto& dof_basis_pair = side_pb.getProvidedDOFs();
    Teuchos::RCP<panzer::PureBasis> basis;
    for (auto it = dof_basis_pair.begin(); it != dof_basis_pair.end(); ++it)
    {
        if (it->first == dof_name)
            basis = it->second;
    }
    return panzer::basisIRLayout(basis, *_ir);
}

//---------------------------------------------------------------------------//
// Register degree of freedom and gradients for a variable with name `dof_name`
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerDOFsGradient(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const std::string& dof_name) const
{
    // Degree of freedom (DOF)
    const auto basis_layout = this->getIntegrationBasis(side_pb, dof_name);
    Teuchos::ParameterList dof_params;
    dof_params.set<std::string>("Name", dof_name);
    dof_params.set<Teuchos::RCP<panzer::BasisIRLayout>>("Basis", basis_layout);
    dof_params.set<Teuchos::RCP<panzer::IntegrationRule>>("IR", _ir);
    const auto dof_op
        = Teuchos::rcp(new panzer::DOF<EvalType, panzer::Traits>(dof_params));
    this->template registerEvaluator<EvalType>(fm, dof_op);

    // Gradient
    dof_params.set<std::string>("Gradient Name", "GRAD_" + dof_name);
    const auto gd_op = Teuchos::rcp(
        new panzer::DOFGradient<EvalType, panzer::Traits>(dof_params));
    this->template registerEvaluator<EvalType>(fm, gd_op);
}

//---------------------------------------------------------------------------//
// Register side nornal evaluator
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerSideNormals(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb) const
{
    std::stringstream normal_params_name;
    normal_params_name << "Side Normal:" << side_pb.cellData().side();
    Teuchos::ParameterList normal_params(normal_params_name.str());
    normal_params.set<std::string>("Name", "Side Normal");
    normal_params.set<int>("Side ID", side_pb.cellData().side());
    normal_params.set<Teuchos::RCP<panzer::IntegrationRule>>("IR", _ir);
    normal_params.set<bool>("Normalize", true);
    const auto normal_op = Teuchos::rcp(
        new panzer::Normals<EvalType, panzer::Traits>(normal_params));
    this->template registerEvaluator<EvalType>(fm, normal_op);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerConvectionTypeFluxOperator(
    std::pair<const std::string, std::string> dof_eq_pair,
    std::unordered_map<std::string, std::vector<std::string>>& eq_res_map,
    const std::string& closure_name,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const double& multiplier) const
{
    // Local variables
    const std::string eq_name = dof_eq_pair.first;
    const std::string dof_name = dof_eq_pair.second;
    const std::string normal_dot_name = "Normal Dot Flux " + eq_name;

    // Register dot product evaluator
    const std::string flux_name = "BOUNDARY_" + closure_name + "_FLUX_"
                                  + eq_name;
    const auto normal_dot_flux_op
        = panzer::buildEvaluator_DotProduct<EvalType, panzer::Traits>(
            normal_dot_name, *_ir, "Side Normal", flux_name);
    this->template registerEvaluator<EvalType>(fm, normal_dot_flux_op);

    // Convective flux integral.
    const auto basis_layout = this->getIntegrationBasis(side_pb, dof_name);
    std::string residual_name = closure_name + "_BOUNDARY_RESIDUAL_" + eq_name;
    const auto convective_flux_op = Teuchos::rcp(
        new panzer::Integrator_BasisTimesScalar<EvalType, panzer::Traits>(
            panzer::EvaluatorStyle::EVALUATES,
            residual_name,
            normal_dot_name,
            *basis_layout,
            *_ir,
            multiplier));
    this->template registerEvaluator<EvalType>(fm, convective_flux_op);

    // Add residual name to `eq_res_map`
    eq_res_map[eq_name].push_back(residual_name);
}

//---------------------------------------------------------------------------//
// Register closure models used for the symmetric penalty method
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerPenaltyAndViscousGradientOperator(
    std::pair<const std::string, std::string> dof_eq_pair,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const Teuchos::ParameterList& user_params) const
{
    // Local variables
    const std::string eq_name = dof_eq_pair.first;
    const std::string dof_name = dof_eq_pair.second;
    const auto basis_layout = this->getIntegrationBasis(side_pb, dof_name);

    // Create viscous penalty parameter.
    const auto viscous_penalty_op
        = Teuchos::rcp(new ViscousPenaltyParameter<EvalType, panzer::Traits>(
            *_ir, *(basis_layout->getBasis()), dof_name, user_params));
    this->template registerEvaluator<EvalType>(fm, viscous_penalty_op);

    // Create boundary gradients.
    const auto viscous_gradient_op = Teuchos::rcp(
        new ViscousGradient<EvalType, panzer::Traits>(*_ir, dof_name));
    this->template registerEvaluator<EvalType>(fm, viscous_gradient_op);
}

//---------------------------------------------------------------------------//
// Register Laplace-type operators
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerViscousTypeFluxOperator(
    std::pair<const std::string, std::string> dof_eq_pair,
    std::unordered_map<std::string, std::vector<std::string>>& eq_res_map,
    const std::string closure_name,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const double& multiplier) const
{
    const std::string eq_name = dof_eq_pair.first;
    const std::string dof_name = dof_eq_pair.second;
    const auto basis_layout = this->getIntegrationBasis(side_pb, dof_name);

    // Symmetric interior penalty method residual 1.
    // FIXME: The dot product evaluator is not on the device in
    // Trilinos 13.0.1
    std::string normal_dot_viscous_name = "Normal Dot " + closure_name
                                          + " Flux " + eq_name;
    std::string flux_name = "BOUNDARY_" + closure_name + "_FLUX_" + eq_name;
    auto normal_dot_viscous_flux_op
        = panzer::buildEvaluator_DotProduct<EvalType, panzer::Traits>(
            normal_dot_viscous_name, *_ir, "Side Normal", flux_name);
    this->template registerEvaluator<EvalType>(fm, normal_dot_viscous_flux_op);

    std::string bnd_resid = closure_name + "_BOUNDARY_RESIDUAL_" + eq_name;
    auto bnd_op = Teuchos::rcp(
        new panzer::Integrator_BasisTimesScalar<EvalType, panzer::Traits>(
            panzer::EvaluatorStyle::EVALUATES,
            bnd_resid,
            normal_dot_viscous_name,
            *basis_layout,
            *_ir,
            -multiplier));
    this->template registerEvaluator<EvalType>(fm, bnd_op);
    eq_res_map[eq_name].push_back(bnd_resid);

    // Symmetric interior penalty method residual 2.
    std::string penalty_bnd_resid = "PENALTY_BOUNDARY_RESIDUAL_" + eq_name;
    auto bnd_penalty_op = Teuchos::rcp(
        new Integrator::BoundaryGradBasisDotVector<EvalType, panzer::Traits>(
            panzer::EvaluatorStyle::EVALUATES,
            penalty_bnd_resid,
            "PENALTY_BOUNDARY_" + closure_name + "_FLUX_" + eq_name,
            *basis_layout,
            *_ir,
            -multiplier));
    this->template registerEvaluator<EvalType>(fm, bnd_penalty_op);
    eq_res_map[eq_name].push_back(penalty_bnd_resid);

    // Symmetric interior penalty method residual 3.
    // FIXME: The dot product evaluator is not on the device in
    // Trilinos 13.0.1
    std::string normal_dot_scaled_penalty_viscous_name
        = "Normal Dot Scaled Penalty " + closure_name + " Flux " + eq_name;
    std::string scaled_penalty_flux_name = "SYMMETRY_BOUNDARY_" + closure_name
                                           + "_FLUX_" + eq_name;
    auto normal_dot_scaled_penalty_viscous_flux_op
        = panzer::buildEvaluator_DotProduct<EvalType, panzer::Traits>(
            normal_dot_scaled_penalty_viscous_name,
            *_ir,
            "Side Normal",
            scaled_penalty_flux_name);
    this->template registerEvaluator<EvalType>(
        fm, normal_dot_scaled_penalty_viscous_flux_op);

    std::string scaled_penalty_bnd_resid = "SYMMETRY_BOUNDARY_RESIDUAL_"
                                           + eq_name;
    auto scaled_bnd_penalty_op = Teuchos::rcp(
        new panzer::Integrator_BasisTimesScalar<EvalType, panzer::Traits>(
            panzer::EvaluatorStyle::EVALUATES,
            scaled_penalty_bnd_resid,
            normal_dot_scaled_penalty_viscous_name,
            *basis_layout,
            *_ir,
            multiplier));
    this->template registerEvaluator<EvalType>(fm, scaled_bnd_penalty_op);
    eq_res_map[eq_name].push_back(scaled_penalty_bnd_resid);
}

//---------------------------------------------------------------------------//
// Register residuals collected in `eq_res_map` for each equation.
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerResidual(
    std::pair<const std::string, std::string> dof_eq_pair,
    std::unordered_map<std::string, std::vector<std::string>>& eq_res_map,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb) const
{
    // Local variables
    const std::string eq_name = dof_eq_pair.first;
    const std::string dof_name = dof_eq_pair.second;
    const auto basis_layout = this->getIntegrationBasis(side_pb, dof_name);

    // Initialize residual vector
    auto residuals = Teuchos::rcp(new std::vector<std::string>);
    for (auto& res : eq_res_map[eq_name])
        residuals->push_back(res);

    // Register residuals
    Teuchos::ParameterList sum_params;
    sum_params.set("Sum Name", "BOUNDARY_RESIDUAL_" + eq_name);
    sum_params.set("Values Names", residuals);
    sum_params.set("Data Layout", basis_layout->getBasis()->functional);
    auto sum_op = Teuchos::rcp(
        new panzer::SumStatic<EvalType, panzer::Traits, panzer::Cell, panzer::BASIS>(
            sum_params));
    this->template registerEvaluator<EvalType>(fm, sum_op);
}

//---------------------------------------------------------------------------//
// Register and scatter residual for each equation
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerScatterOperator(
    std::pair<const std::string, std::string> dof_eq_pair,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const panzer::LinearObjFactory<panzer::Traits>& lof) const
{
    const auto& dof_basis_pair = side_pb.getProvidedDOFs();
    const std::string& eq_name = dof_eq_pair.first;
    const std::string& dof_name = dof_eq_pair.second;

    Teuchos::RCP<const panzer::PureBasis> basis;
    for (auto it = dof_basis_pair.begin(); it != dof_basis_pair.end(); ++it)
    {
        if (it->first == dof_name)
            basis = it->second;
    }

    std::string residual_name = "BOUNDARY_RESIDUAL_" + eq_name;
    Teuchos::ParameterList p("Scatter: " + residual_name + " to " + eq_name);

    std::string scatter_field_name
        = "Dummy Scatter: " + this->m_bc.identifier() + residual_name;
    p.set("Scatter Name", scatter_field_name);
    p.set("Basis", basis);

    auto residual_names = Teuchos::rcp(new std::vector<std::string>);
    residual_names->push_back(residual_name);
    p.set("Dependent Names", residual_names);

    auto names_map = Teuchos::rcp(new std::map<std::string, std::string>);
    names_map->insert(
        std::pair<std::string, std::string>(residual_name, dof_name));
    p.set("Dependent Map", names_map);

    auto scatter_op = lof.buildScatter<EvalType>(p);

    this->template registerEvaluator<EvalType>(fm, scatter_op);

    // Require variables
    auto dummy_layout = Teuchos::rcp(new PHX::MDALayout<panzer::Dummy>(0));
    PHX::Tag<typename EvalType::ScalarT> tag(scatter_field_name, dummy_layout);
    fm.template requireField<EvalType>(tag);
}
//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYCONDITION_BOUNDARYFLUXBASE_IMPL_HPP

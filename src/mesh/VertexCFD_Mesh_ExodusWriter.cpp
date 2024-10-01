#include "VertexCFD_Mesh_ExodusWriter.hpp"

#include "utils/VertexCFD_Utils_VectorizeOutputFieldNames.hpp"

#include <Panzer_String_Utilities.hpp>

namespace VertexCFD
{
namespace Mesh
{
//---------------------------------------------------------------------------//
void ExodusWriter::add_mesh_outputs(const Teuchos::ParameterList& params,
                                    const OutputType output_type,
                                    const OutputLocation output_location)
{
    for (const auto& param : params)
    {
        const auto& block_id = param.first;
        const auto tokens = VectorizeOutputFieldNames::tokenizeParameter(param);
        for (const auto& field : tokens)
        {
            if (output_location == OutputLocation::Node)
            {
                // nodal scalar
                _mesh->addSolutionField(field, block_id);
            }
            else if (output_type == OutputType::Scalar)
            {
                // cell scalar
                _mesh->addCellField(field, block_id);
            }
            else
            {
                // cell vector
                constexpr char dim_name[3] = {'X', 'Y', 'Z'};
                for (std::size_t dim = 0; dim < _mesh->getDimension(); ++dim)
                {
                    _mesh->addCellField(field + dim_name[dim], block_id);
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
ExodusWriter::ExodusWriter(
    const Teuchos::RCP<panzer_stk::STK_Interface>& mesh,
    const Teuchos::RCP<const panzer::GlobalIndexer>& dof_manager,
    const Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits>>& lof,
    const Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits>>& response_library,
    const Teuchos::ParameterList& output_params)
    : _mesh(mesh)
    , _dof_manager(dof_manager)
    , _lof(lof)
    , _response_library(response_library)
{
    add_mesh_outputs(output_params.sublist("Cell Average Quantities"),
                     OutputType::Scalar,
                     OutputLocation::Cell);
    add_mesh_outputs(output_params.sublist("Cell Average Vectors"),
                     OutputType::Vector,
                     OutputLocation::Cell);
    add_mesh_outputs(output_params.sublist("Cell Quantities"),
                     OutputType::Scalar,
                     OutputLocation::Cell);
    add_mesh_outputs(output_params.sublist("Nodal Quantities"),
                     OutputType::Scalar,
                     OutputLocation::Node);

    _mesh->setupExodusFile(
        output_params.template get<std::string>("Exodus Output File"));

    std::vector<std::string> element_blocks;
    _mesh->getElementBlockNames(element_blocks);
    panzer_stk::RespFactorySolnWriter_Builder response_builder;
    response_builder.mesh = mesh;

    _response_library->addResponse(
        "Main Field Output", element_blocks, response_builder);
}

//---------------------------------------------------------------------------//
void ExodusWriter::writeSolution(
    const Teuchos::RCP<const Thyra::VectorBase<double>>& x,
    const Teuchos::RCP<const Thyra::VectorBase<double>>& x_dot,
    const double time,
    const double time_step)
{
    panzer::AssemblyEngineInArgs in_args;
    in_args.container_ = _lof->buildLinearObjContainer();
    in_args.ghostedContainer_ = _lof->buildGhostedLinearObjContainer();
    in_args.alpha = 0.0;
    in_args.beta = 1.0;
    in_args.time = time;
    in_args.step_size = time_step;
    in_args.evaluate_transient_terms = true;

    _lof->initializeGhostedContainer(
        panzer::LinearObjContainer::X | panzer::LinearObjContainer::DxDt,
        *(in_args.ghostedContainer_));

    auto thyra_container
        = Teuchos::rcp_dynamic_cast<panzer::ThyraObjContainer<double>>(
            in_args.container_, true);
    thyra_container->set_x_th(
        Teuchos::rcp_const_cast<Thyra::VectorBase<double>>(x));
    thyra_container->set_dxdt_th(
        Teuchos::rcp_const_cast<Thyra::VectorBase<double>>(x_dot));

    _response_library->addResponsesToInArgs<panzer::Traits::Residual>(in_args);
    _response_library->evaluate<panzer::Traits::Residual>(in_args);

    _mesh->writeToExodus(time);
}

//---------------------------------------------------------------------------//

} // end namespace Mesh
} // end namespace VertexCFD

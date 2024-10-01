#include "VertexCFD_PhysicsManager.hpp"
#include "boundary_conditions/VertexCFD_BCStrategy_Factory.hpp"
#include "closure_models/VertexCFD_ClosureModelFactory_TemplateBuilder.hpp"
#include "equation_sets/VertexCFD_EquationSet_Factory.hpp"
#include "linear_solvers/VertexCFD_LinearSolvers_LOWSFactoryBuilder.hpp"

#include <PanzerAdaptersSTK_config.hpp>
#include <Panzer_BlockedEpetraLinearObjFactory.hpp>
#include <Panzer_BlockedTpetraLinearObjFactory.hpp>
#include <Panzer_ClosureModel_Factory_TemplateManager.hpp>
#include <Panzer_DOFManagerFactory.hpp>
#include <Panzer_ElementBlockIdToPhysicsIdMap.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_STK_IOClosureModel_Factory_TemplateBuilder.hpp>
#include <Panzer_STK_WorksetFactory.hpp>
#include <Panzer_String_Utilities.hpp>

#include <map>
#include <string>
#include <vector>

namespace VertexCFD
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
PhysicsManager::PhysicsManager(
    const std::integral_constant<int, NumSpaceDim>&,
    const Teuchos::RCP<Parameter::ParameterDatabase>& parameter_db,
    const Teuchos::RCP<MeshManager>& mesh_manager,
    const double initial_time)
    : _parameter_db(parameter_db)
    , _mesh_manager(mesh_manager)
    , _t_init(initial_time)
    , _global_data(panzer::createGlobalData())
    , _integration_order(-1)
{
    // Initialize 'num_space_dim' with template value
    constexpr int num_space_dim = NumSpaceDim;

    // Get physics communicator.
    auto comm = _mesh_manager->comm();

    // Get user parameters.
    auto user_params = _parameter_db->userParameters();

    // Map element blocks to physics blocks.
    auto mesh = _mesh_manager->mesh();
    const auto block_mapping_params = _parameter_db->blockMappingParameters();
    std::map<std::string, std::string> block_ids_to_physics_ids;
    panzer::buildBlockIdToPhysicsIdMap(block_ids_to_physics_ids,
                                       *block_mapping_params);
    std::map<std::string, Teuchos::RCP<const shards::CellTopology>>
        block_ids_to_cell_topo;
    for (auto itr(block_ids_to_physics_ids.begin());
         itr != block_ids_to_physics_ids.end();
         ++itr)
    {
        block_ids_to_cell_topo[itr->first] = mesh->getCellTopology(itr->first);
    }

    // Get physics parameters.
    const auto physics_params = _parameter_db->physicsParameters();

    // Build the physics blocks. Create defaults for some options that may be
    // overwritten by inputs.
    const int workset_size = user_params->get<int>("Workset Size");
    const int default_integration_order = 2;
    const bool build_transient_support = true;
    const std::vector<std::string> tangent_param_names;
    _eq_set_factory = Teuchos::rcp(new VertexCFD::EquationSet::Factory);
    panzer::buildPhysicsBlocks(block_ids_to_physics_ids,
                               block_ids_to_cell_topo,
                               physics_params,
                               default_integration_order,
                               workset_size,
                               _eq_set_factory,
                               _global_data,
                               build_transient_support,
                               _physics_blocks,
                               tangent_param_names);

    // FIXME: Everything breaks if our integration order is not consistent, so
    //        just extract the actaul order from the frist physics block for
    //        use everywhere else.
    _integration_order
        = _physics_blocks.at(0)->getIntegrationRules().cbegin()->first;

    // Generate field data to complete the mesh.
    const int num_physics_blocks = _physics_blocks.size();
    for (int i = 0; i < num_physics_blocks; ++i)
    {
        // Get a unique list of the fields names needed by the physics and add
        // them to the database.
        auto pb = _physics_blocks[i];
        const auto& block_fields = pb->getProvidedDOFs();
        std::set<panzer::StrPureBasisPair, panzer::StrPureBasisComp> field_names(
            block_fields.begin(), block_fields.end());
        for (auto field = field_names.begin(); field != field_names.end();
             ++field)
        {
            mesh->addSolutionField(field->first, pb->elementBlockID());
        }
    }
    _mesh_manager->completeMeshConstruction();
    // Generate connectivity and DOF managers.
    auto conn_manager = _mesh_manager->connectivityManager();
    panzer::DOFManagerFactory dof_manager_factory;
    _dof_manager = dof_manager_factory.buildGlobalIndexer(
        Teuchos::opaqueWrapper(Teuchos::getRawMpiComm(*comm)),
        _physics_blocks,
        conn_manager);

    // Toggle on linear algebra type to construct linear object factory
    const std::string lin_alg_type
        = user_params->get<std::string>("Linear Algebra Type", "Tpetra");
    if (lin_alg_type == "Tpetra")
    {
        _linear_object_factory = Teuchos::rcp(
            new panzer::TpetraLinearObjFactory<panzer::Traits,
                                               double,
                                               int,
                                               panzer::GlobalOrdinal>(
                comm, _dof_manager));
    }
    else if (lin_alg_type == "Epetra")
    {
        _linear_object_factory = Teuchos::rcp(
            new panzer::BlockedEpetraLinearObjFactory<panzer::Traits, int>(
                comm, _dof_manager));
    }
    else
    {
        throw std::runtime_error(
            "Invalid linear algebra type. Valid options are 'Tpetra' and "
            "'Epetra'");
    }

    // Linear solver factory.
    auto linear_solver_params = parameter_db->linearSolverParameters();
    auto lows_factory = VertexCFD::LinearSolvers::LOWSFactoryBuilder::buildLOWS(
        linear_solver_params);

    // Create boundary conditions.
    auto bc_params = _parameter_db->boundaryConditionParameters();
    panzer::buildBCs(_boundary_conditions, *bc_params, _global_data);

    // Create model evaluator.
    _model_evaluator = Teuchos::rcp(
        new panzer::ModelEvaluator<double>(_linear_object_factory,
                                           lows_factory,
                                           _global_data,
                                           build_transient_support,
                                           _t_init));

    // Create additional factories.
    _bc_factory = Teuchos::rcp(
        new VertexCFD::BoundaryCondition::Factory<num_space_dim>());
    VertexCFD::ClosureModel::FactoryTemplateBuilder<num_space_dim> cm_builder;
    _cm_factory = Teuchos::rcp(
        new panzer::ClosureModelFactory_TemplateManager<panzer::Traits>());
    _cm_factory->buildObjects(cm_builder);
}

//---------------------------------------------------------------------------//
// Explicit instantiation of the constructor 'PhysicsManager'
template PhysicsManager::PhysicsManager(
    const std::integral_constant<int, 2>&,
    const Teuchos::RCP<Parameter::ParameterDatabase>& parameter_db,
    const Teuchos::RCP<MeshManager>& mesh_manager,
    const double initial_time);

template PhysicsManager::PhysicsManager(
    const std::integral_constant<int, 3>&,
    const Teuchos::RCP<Parameter::ParameterDatabase>& parameter_db,
    const Teuchos::RCP<MeshManager>& mesh_manager,
    const double initial_time);

//---------------------------------------------------------------------------//
// Add a scalar parameter and an initial value for the parameter.
int PhysicsManager::addScalarParameter(const std::string& name,
                                       const double value)
{
    int param_id = _model_evaluator->addParameter(name, value);
    _parameter_indices.emplace(name, param_id);
    return param_id;
}

//---------------------------------------------------------------------------//
// Get the index of a parameter with the given name.
int PhysicsManager::getParameterIndex(const std::string& name) const
{
    auto iter = _parameter_indices.find(name);
    if (iter == _parameter_indices.end())
    {
        throw std::runtime_error("Parameter not found");
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
void PhysicsManager::setupModel()
{
    // Create worksets.
    auto user_params = _parameter_db->userParameters();
    auto mesh = _mesh_manager->mesh();
    auto workset_factory = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh));
    _workset_container = Teuchos::rcp(new panzer::WorksetContainer);
    _workset_container->setFactory(workset_factory);
    const int num_physics_blocks = _physics_blocks.size();
    for (int i = 0; i < num_physics_blocks; ++i)
    {
        auto pb = _physics_blocks[i];
        _workset_container->setNeeds(pb->elementBlockID(),
                                     pb->getWorksetNeeds());
    }
    const int workset_size = user_params->get<int>("Workset Size");
    _workset_container->setWorksetSize(workset_size);
    _workset_container->setGlobalIndexer(_dof_manager);

    // Turn on shared memory versions of Panzer kernels when available
    panzer::HP::inst().setUseSharedMemory(true, true);

    // Setup model.
    auto closure_params = _parameter_db->closureModelParameters();
    const bool write_graph = user_params->get<bool>("Output Graph");
    user_params->set<const Teuchos::RCP<MeshManager>>("MeshManager",
                                                      _mesh_manager);
    _model_evaluator->setupModel(_workset_container,
                                 _physics_blocks,
                                 _boundary_conditions,
                                 *_eq_set_factory,
                                 *_bc_factory,
                                 *_cm_factory,
                                 *_cm_factory,
                                 *closure_params,
                                 *user_params,
                                 write_graph,
                                 "");

    // Add scalar parameters.
    auto scalar_params = _parameter_db->scalarParameters();
    if (Teuchos::nonnull(scalar_params))
    {
        for (auto sp = scalar_params->begin(); sp != scalar_params->end(); ++sp)
        {
            auto list = scalar_params->sublist(sp->first);
            addScalarParameter(list.get<std::string>("Name"),
                               list.get<double>("Nominal Value"));
        }
    }
}

//---------------------------------------------------------------------------//
Teuchos::RCP<MeshManager> PhysicsManager::meshManager() const
{
    return _mesh_manager;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer::GlobalData> PhysicsManager::globalData() const
{
    return _global_data;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer::EquationSetFactory>
PhysicsManager::equationSetFactory() const
{
    return _eq_set_factory;
}

//---------------------------------------------------------------------------//
const std::vector<Teuchos::RCP<panzer::PhysicsBlock>>&
PhysicsManager::physicsBlocks() const
{
    return _physics_blocks;
}

//---------------------------------------------------------------------------//
int PhysicsManager::integrationOrder() const
{
    return _integration_order;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer::GlobalIndexer> PhysicsManager::dofManager() const
{
    return _dof_manager;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits>>
PhysicsManager::linearObjectFactory() const
{
    return _linear_object_factory;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer::WorksetContainer> PhysicsManager::worksetContainer() const
{
    return _workset_container;
}

//---------------------------------------------------------------------------//
const std::vector<panzer::BC>& PhysicsManager::boundaryConditions() const
{
    return _boundary_conditions;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer::BCStrategyFactory>
PhysicsManager::boundaryConditionFactory() const
{
    return _bc_factory;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer::ClosureModelFactory_TemplateManager<panzer::Traits>>
PhysicsManager::closureModelFactory() const
{
    return _cm_factory;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer::ModelEvaluator<double>>
PhysicsManager::modelEvaluator() const
{
    return _model_evaluator;
}

//---------------------------------------------------------------------------//

} // end namespace VertexCFD

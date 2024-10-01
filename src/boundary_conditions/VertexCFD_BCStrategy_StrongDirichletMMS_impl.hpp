#ifndef VERTEXCFD_BOUNDARYCONDITION_STORNGDIRICHLETMMS_IMPL_HPP
#define VERTEXCFD_BOUNDARYCONDITION_STORNGDIRICHLETMMS_IMPL_HPP

#include <Panzer_Copy.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType>
StrongDirichletMMS<EvalType>::StrongDirichletMMS(
    const panzer::BC& bc, const Teuchos::RCP<panzer::GlobalData>& global_data)
    : panzer::BCStrategy_Dirichlet_DefaultImpl<EvalType>(bc, global_data)
{
    if (this->m_bc.strategy() != "StrongDirichletMMS")
    {
        throw std::runtime_error("StrongDirichletMMS BC name incorrect");
    }
}

//---------------------------------------------------------------------------//
template<class EvalType>
void StrongDirichletMMS<EvalType>::setup(const panzer::PhysicsBlock& side_pb,
                                         const Teuchos::ParameterList&)
{
    _dofs = side_pb.getProvidedDOFs();
    for (auto& dof : _dofs)
    {
        this->addDOF(dof.first);
        this->addTarget("StrongDirichletMMS_" + dof.first, dof.first);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType>
void StrongDirichletMMS<EvalType>::buildAndRegisterEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock&,
    const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>&,
    const Teuchos::ParameterList&,
    const Teuchos::ParameterList&) const
{
    for (auto& dof : _dofs)
    {
        Teuchos::ParameterList p("BC MMS Strong Dirichlet");
        p.set("Source Name", dof.first);
        p.set("Destination Name", "StrongDirichletMMS_" + dof.first);
        p.set("Data Layout", dof.second->functional);
        auto op = Teuchos::rcp(new panzer::Copy<EvalType, panzer::Traits>(p));
        this->template registerEvaluator<EvalType>(fm, op);
    }
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYCONDITION_STORNGDIRICHLETMMS_IMPL_HPP

#ifndef VERTEXCFD_BOUNDARYCONDITION_FACTORY_HPP
#define VERTEXCFD_BOUNDARYCONDITION_FACTORY_HPP

#include "VertexCFD_BCStrategy_IncompressibleBoundaryFlux.hpp"
#include "VertexCFD_BCStrategy_StrongDirichletMMS.hpp"

#include <Panzer_BCStrategy_Factory.hpp>
#include <Panzer_BCStrategy_TemplateManager.hpp>
#include <Panzer_GlobalData.hpp>
#include <Panzer_Traits.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
namespace Impl
{
template<template<class, int... Dims> class Strategy, int... Dims>
class TemplateBuilder
{
  public:
    TemplateBuilder(const panzer::BC& bc,
                    const Teuchos::RCP<panzer::GlobalData>& global_data)
        : _bc(bc)
        , _global_data(global_data)
    {
    }

    template<class EvalT>
    Teuchos::RCP<panzer::BCStrategyBase> build() const
    {
        Strategy<EvalT, Dims...>* ptr
            = new Strategy<EvalT, Dims...>(_bc, _global_data);
        return Teuchos::rcp(ptr);
    }

  private:
    const panzer::BC& _bc;
    const Teuchos::RCP<panzer::GlobalData> _global_data;
};
} // namespace Impl

//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class Factory : public panzer::BCStrategyFactory
{
  public:
    Teuchos::RCP<panzer::BCStrategy_TemplateManager<panzer::Traits>>
    buildBCStrategy(
        const panzer::BC& bc,
        const Teuchos::RCP<panzer::GlobalData>& global_data) const override
    {
        auto template_manager = Teuchos::rcp(
            new panzer::BCStrategy_TemplateManager<panzer::Traits>{});

        const auto& bc_strategy = bc.strategy();

        if (bc_strategy == "IncompressibleBoundaryFlux")
        {
            Impl::TemplateBuilder<IncompressibleBoundaryFlux, NumSpaceDim> builder(
                bc, global_data);
            template_manager->buildObjects(builder);
        }
        else if (bc_strategy == "StrongDirichletMMS")
        {
            Impl::TemplateBuilder<StrongDirichletMMS> builder(bc, global_data);
            template_manager->buildObjects(builder);
        }
        else
        {
            throw std::runtime_error("BC strategy not valid");
        }

        // Return the manager.
        return template_manager;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYCONDITION_FACTORY_HPP

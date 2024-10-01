#ifndef VERTEXCFD_FULLINDUCTIONBOUNDARYSTATE_FACTORY_HPP
#define VERTEXCFD_FULLINDUCTIONBOUNDARYSTATE_FACTORY_HPP

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"

#include "full_induction_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_FullInductionFixed.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MagneticPressure.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticField.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ExternalMagneticField.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class FullInductionBoundaryStateFactory
{
  public:
    static constexpr int num_space_dim = NumSpaceDim;

    static std::vector<Teuchos::RCP<PHX::Evaluator<Traits>>>
    create(const panzer::IntegrationRule& ir,
           const Teuchos::ParameterList& bc_params,
           const Teuchos::ParameterList& user_params,
           const MHDProperties::FullInductionMHDProperties& mhd_props)
    {
        // Evaluator vector to return
        std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>> evaluators;

        // Add total magnetic field, magnetic pressure, and resistivity
        // closures
        const auto ext_magn_field_op = Teuchos::rcp(
            new ClosureModel::ExternalMagneticField<EvalType, panzer::Traits>(
                ir, user_params));
        evaluators.push_back(ext_magn_field_op);

        const auto tot_magn_field_op = Teuchos::rcp(
            new ClosureModel::
                TotalMagneticField<EvalType, panzer::Traits, num_space_dim>(
                    ir, "BOUNDARY_"));
        evaluators.push_back(tot_magn_field_op);

        const auto magn_press_op = Teuchos::rcp(
            new ClosureModel::MagneticPressure<EvalType, panzer::Traits>(
                ir, mhd_props));
        evaluators.push_back(magn_press_op);

        const auto resistivity_op = Teuchos::rcp(
            new ClosureModel::ConstantScalarField<EvalType, panzer::Traits>(
                ir, "resistivity", mhd_props.resistivity()));
        evaluators.push_back(resistivity_op);

        // Loop over boundary conditions found in input file for the
        // induced magnetic field and scalar magnetic potential
        bool found_model = false;
        if (bc_params.isType<std::string>("Type"))
        {
            const auto bc_type = bc_params.get<std::string>("Type");
            if (bc_type == "Fixed")
            {
                const auto state = Teuchos::rcp(
                    new FullInductionFixed<EvalType, Traits, num_space_dim>(
                        ir, bc_params, mhd_props));
                evaluators.push_back(state);
                found_model = true;
            }

            // Error message if model not found
            if (!found_model)
            {
                std::string msg = "\n\nBoundary state " + bc_type
                                  + " failed to build.\n";
                msg += "The boundary conditions implemented in VERTEX-CFD\n";
                msg += "for the full induction equations are:\n";
                msg += "Fixed,\n";
                msg += "\n";
                throw std::runtime_error(msg);
            }
        }

        // Return vector of evaluators
        return evaluators;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_FULLINDUCTIONBOUNDARYSTATE_FACTORY_HPP

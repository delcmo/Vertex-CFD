#ifndef VERTEXCFD_INITIALCONDITION_MHDVORTEXPROBLEM_HPP
#define VERTEXCFD_INITIALCONDITION_MHDVORTEXPROBLEM_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_PureBasis.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace InitialCondition
{
template<class EvalType, class Traits, int NumSpaceDim>
class MHDVortexProblem : public panzer::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    MHDVortexProblem(const Teuchos::ParameterList& params,
                     const panzer::PureBasis& basis);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS> _lagrange_pressure;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS>,
                  num_space_dim>
        _velocity;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS>,
                  num_space_dim>
        _induced_magnetic_field;

  private:
    std::string _basis_name;
    int _basis_index;
    PHX::MDField<double, panzer::Cell, panzer::BASIS, panzer::Dim> _basis_coords;

    Kokkos::Array<double, 2> _vel_0;
    Kokkos::Array<double, 2> _xy_0;
};

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#include "VertexCFD_InitialCondition_MHDVortexProblem_impl.hpp"

#endif // end VERTEXCFD_INITIALCONDITION_MHDVORTEXPROBLEM_HPP

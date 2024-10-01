#ifndef VERTEXCFD_INITIALCONDITION_METHODMANUFACTUREDSOLUTION_HPP
#define VERTEXCFD_INITIALCONDITION_METHODMANUFACTUREDSOLUTION_HPP

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
class MethodManufacturedSolution
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;
    static constexpr int num_coeff = 2 * (num_space_dim + 1);

    MethodManufacturedSolution(const panzer::PureBasis& basis);

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
    PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS> _temperature;

  private:
    std::string _basis_name;
    int _basis_index;
    PHX::MDField<double, panzer::Cell, panzer::BASIS, panzer::Dim> _basis_coords;

    Kokkos::Array<double, num_coeff> _phi_coeff;
    Kokkos::Array<Kokkos::Array<double, num_coeff>, num_space_dim> _vel_coeff;
    Kokkos::Array<double, num_coeff> _T_coeff;
};

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#include "VertexCFD_InitialCondition_MethodManufacturedSolution_impl.hpp"

#endif // end VERTEXCFD_INITIALCONDITION_METHODMANUFACTUREDSOLUTION_HPP

#ifndef VERTEXCFD_CLOSURE_MAGNETICCORRECTIONDAMPINGSOURCE_HPP
#define VERTEXCFD_CLOSURE_MAGNETICCORRECTIONDAMPINGSOURCE_HPP

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Damping term for the magnetic correction potential field
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class MagneticCorrectionDampingSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    MagneticCorrectionDampingSource(
        const panzer::IntegrationRule& ir,
        const MHDProperties::FullInductionMHDProperties& mhd_props);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _damping_potential_source;

  private:
    double _alpha;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _scalar_magnetic_potential;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_MAGNETICCORRECTIONDAMPINGSOURCE_HPP

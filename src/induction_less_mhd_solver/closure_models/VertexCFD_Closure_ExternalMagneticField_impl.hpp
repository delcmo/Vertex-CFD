#ifndef VERTEXCFD_CLOSURE_EXTERNALMAGNETICFIELD_IMPL_HPP
#define VERTEXCFD_CLOSURE_EXTERNALMAGNETICFIELD_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ExternalMagneticField<EvalType, Traits>::ExternalMagneticField(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& user_params)
{
    // Get external magnetic vector
    const auto ext_magn_vct
        = user_params.get<Teuchos::Array<double>>("External Magnetic Field");
    for (int dim = 0; dim < field_size; ++dim)
        _ext_magn_vct[dim] = ext_magn_vct[dim];

    // Evaluated fields
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _ext_magn_field, "external_magnetic_field_");

    this->setName("Electric Potential External Magnetic Field");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ExternalMagneticField<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ExternalMagneticField<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _ext_magn_field[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < field_size; ++dim)
                _ext_magn_field[dim](cell, point) = _ext_magn_vct[dim];
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_EXTERNALMAGNETICFIELD_IMPL_HPP

#ifndef VERTEXCFD_UTILS_SCALARTOVECTOR_IMPL_HPP
#define VERTEXCFD_UTILS_SCALARTOVECTOR_IMPL_HPP

#include "VertexCFD_Utils_ScalarToVector.hpp"

#include <Phalanx_DataLayout_MDALayout.hpp>

#include <string>

namespace VertexCFD
{
namespace Utils
{
//---------------------------------------------------------------------------//
template<typename EvalType, typename DimTag>
ScalarToVector<EvalType, DimTag>::ScalarToVector(
    const panzer::IntegrationRule& ir,
    const std::string& field_name,
    const int num_scalars,
    const bool time_deriv)
{
    // Build scalar and gradient fields
    _scalar_fields.reserve(num_scalars);
    _scalar_grad_fields.reserve(num_scalars);
    std::string name;
    for (int sc = 0; sc < num_scalars; ++sc)
    {
        name = field_name + "_" + std::to_string(sc);
        _scalar_fields.emplace_back(name, ir.dl_scalar);

        name = "GRAD_" + field_name + "_" + std::to_string(sc);
        _scalar_grad_fields.emplace_back(name, ir.dl_vector);
    }

    // Create new data layout for vector component
    Teuchos::RCP<PHX::DataLayout> vector_layout;

    // field is Cell, Point
    vector_layout
        = Teuchos::rcp(new PHX::MDALayout<panzer::Cell, panzer::Point, DimTag>(
            ir.dl_scalar->extent(0), ir.dl_scalar->extent(1), num_scalars));
    _vector_fields = PHX::MDField<ScalarT, panzer::Cell, panzer::Point, DimTag>(
        field_name, vector_layout);

    // GRAD field is Cell, Point, Dim
    vector_layout = Teuchos::rcp(
        new PHX::MDALayout<panzer::Cell, panzer::Point, panzer::Dim, DimTag>(
            ir.dl_vector->extent(0),
            ir.dl_vector->extent(1),
            ir.dl_vector->extent(2),
            num_scalars));
    _vector_grad_fields
        = PHX::MDField<ScalarT, panzer::Cell, panzer::Point, panzer::Dim, DimTag>(
            "GRAD_" + field_name, vector_layout);

    // Add dependent/evaluated fields
    this->addEvaluatedField(_vector_fields);
    this->addEvaluatedField(_vector_grad_fields);

    for (int sc = 0; sc < num_scalars; ++sc)
    {
        this->addDependentField(_scalar_fields[sc]);
        this->addDependentField(_scalar_grad_fields[sc]);
    }

    // Add time-derivative components if requested
    if (time_deriv)
    {
        // Set up scalar fields
        _scalar_dxdt_fields.reserve(num_scalars);
        for (int sc = 0; sc < num_scalars; ++sc)
        {
            name = "DXDT_" + field_name + "_" + std::to_string(sc);
            _scalar_dxdt_fields.emplace_back(name, ir.dl_scalar);
        }

        // Create vector layout/field
        vector_layout = Teuchos::rcp(
            new PHX::MDALayout<panzer::Cell, panzer::Point, DimTag>(
                ir.dl_scalar->extent(0), ir.dl_scalar->extent(1), num_scalars));
        _vector_dxdt_fields
            = PHX::MDField<ScalarT, panzer::Cell, panzer::Point, DimTag>(
                "DXDT_" + field_name, vector_layout);

        // Register fields
        this->addEvaluatedField(_vector_dxdt_fields);
        for (int sc = 0; sc < num_scalars; ++sc)
            this->addDependentField(_scalar_dxdt_fields[sc]);
    }

    this->setName("ScalarToVector");
}

//---------------------------------------------------------------------------//
template<typename EvalType, typename DimTag>
void ScalarToVector<EvalType, DimTag>::evaluateFields(
    typename panzer::Traits::EvalData)
{
    const int num_scalars = _scalar_fields.size();

    // Process scalars sequentially
    for (int sc = 0; sc < num_scalars; ++sc)
    {
        // Copy field
        auto scalar_field_view = _scalar_fields[sc].get_view();
        auto vector_field_view = Kokkos::subview(
            _vector_fields.get_view(), Kokkos::ALL(), Kokkos::ALL(), sc);
        Kokkos::deep_copy(vector_field_view, scalar_field_view);

        if (_scalar_dxdt_fields.size() > 0)
        {
            // Copy DXDT field
            auto scalar_dxdt_field_view = _scalar_dxdt_fields[sc].get_view();
            auto vector_dxdt_field_view
                = Kokkos::subview(_vector_dxdt_fields.get_view(),
                                  Kokkos::ALL(),
                                  Kokkos::ALL(),
                                  sc);
            Kokkos::deep_copy(vector_dxdt_field_view, scalar_dxdt_field_view);
        }

        // Copy GRAD field
        auto scalar_grad_field_view = _scalar_grad_fields[sc].get_view();
        auto vector_grad_field_view
            = Kokkos::subview(_vector_grad_fields.get_view(),
                              Kokkos::ALL(),
                              Kokkos::ALL(),
                              Kokkos::ALL(),
                              sc);
        Kokkos::deep_copy(vector_grad_field_view, scalar_grad_field_view);
    }
}

//---------------------------------------------------------------------------//

} // namespace Utils
} // namespace VertexCFD

#endif // VERTEXCFD_UTILS_SCALARTOVECTOR_IMPL_HPP

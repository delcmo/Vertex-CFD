#ifndef VERTEXCFD_CONSTANTFLUIDPROPERTIES_HPP
#define VERTEXCFD_CONSTANTFLUIDPROPERTIES_HPP

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace FluidProperties
{
//---------------------------------------------------------------------------//
// Constant fluid properties
//---------------------------------------------------------------------------//

class ConstantFluidProperties
{
  public:
    ConstantFluidProperties() = default;
    explicit ConstantFluidProperties(const Teuchos::ParameterList& params)
        : _kinematic_viscosity(params.get<double>("Kinematic viscosity"))
        , _beta(params.get<double>("Artificial compressibility"))
        , _solve_temp(params.get<bool>("Build Temperature Equation"))
        , _build_elec_pot_equ(false)
        , _build_buoyancy(false)
    {
        // Density
        if (params.isType<double>("Density"))
        {
            _density = params.get<double>("Density");
        }
        else
        {
            _density = 1.0;
        }

        // Thermal parameters
        if (_solve_temp)
        {
            _thermal_conductivity = params.get<double>("Thermal conductivity");
            _Cp = params.get<double>("Specific heat capacity");

            // Check for buoyancy bool
            if (params.isType<bool>("Build Buoyancy Source"))
            {
                _build_buoyancy = params.get<bool>("Build Buoyancy Source");
            }
        }
        else
        {
            _thermal_conductivity = std::numeric_limits<double>::quiet_NaN();
            _Cp = std::numeric_limits<double>::quiet_NaN();
        }

        // Buoyancy source term

        if (_build_buoyancy)
        {
            _beta_T = params.get<double>("Expansion coefficient");
            _T_ref = params.get<double>("Reference temperature");
        }
        else
        {
            _beta_T = std::numeric_limits<double>::quiet_NaN();
            _T_ref = std::numeric_limits<double>::quiet_NaN();
        }

        // Electric potential equation
        if (params.isType<bool>("Build Electric Potential Equation"))
        {
            _build_elec_pot_equ
                = params.get<bool>("Build Electric Potential Equation");
        }

        if (_build_elec_pot_equ)
        {
            _sigma = params.get<double>("Electrical conductivity");
        }
        else
        {
            _sigma = std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Constant density
    KOKKOS_INLINE_FUNCTION double constantDensity() const { return _density; }

    // Constant kinematic viscosity
    KOKKOS_INLINE_FUNCTION double constantKinematicViscosity() const
    {
        return _kinematic_viscosity;
    }

    // Constant thermal conductivity
    KOKKOS_INLINE_FUNCTION double constantThermalConductivity() const
    {
        return _thermal_conductivity;
    }

    // Constant heat capacity
    KOKKOS_INLINE_FUNCTION double constantHeatCapacity() const
    {
        return _density * _Cp;
    }

    // Constant electrical conductivity
    KOKKOS_INLINE_FUNCTION double constantElectricalConductivity() const
    {
        return _sigma;
    }

    // Solve temperature equation
    KOKKOS_INLINE_FUNCTION bool solveTemperature() const
    {
        return _solve_temp;
    }

    // Include buoyancy effects
    KOKKOS_INLINE_FUNCTION bool buildBuoyancy() const
    {
        return _build_buoyancy;
    }

    // Expansion coefficient
    KOKKOS_INLINE_FUNCTION double expansionCoefficient() const
    {
        return _beta_T;
    }

    // Reference temperature
    KOKKOS_INLINE_FUNCTION double referenceTemperature() const
    {
        return _T_ref;
    }

    // Artificial compressibility
    KOKKOS_INLINE_FUNCTION double artificialCompressibility() const
    {
        return _beta;
    }

  private:
    double _kinematic_viscosity;
    double _beta;
    double _density;
    double _Cp;
    double _thermal_conductivity;
    double _sigma;
    double _beta_T;
    double _T_ref;
    bool _solve_temp;
    bool _build_elec_pot_equ;
    bool _build_buoyancy;
};

} // namespace FluidProperties
} // namespace VertexCFD

#endif // VERTEXCFD_CONSTANTFLUIDPROPERTIES_HPP

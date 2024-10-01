#ifndef VERTEXCFD_FULLINDUCTIONMHDPROPERTIES_HPP
#define VERTEXCFD_FULLINDUCTIONMHDPROPERTIES_HPP

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace MHDProperties
{
//---------------------------------------------------------------------------//
// Full induction MHD properties
//---------------------------------------------------------------------------//

class FullInductionMHDProperties
{
  public:
    FullInductionMHDProperties() = default;
    explicit FullInductionMHDProperties(const Teuchos::ParameterList& mhd_params)
        : _build_magn_corr(false)
        , _build_resistive_flux(false)
        , _variable_resistivity(false)
        , _mu_0(1.0)
        , _eta(std::numeric_limits<double>::quiet_NaN())
        , _c_h(0.0)
    {
        if (mhd_params.isType<bool>("Build Magnetic Correction "
                                    "Potential Equation"))
        {
            _build_magn_corr = mhd_params.get<bool>(
                "Build Magnetic "
                "Correction Potential "
                "Equation");
        }

        // Vacuum magnetic permeability
        if (mhd_params.isType<double>("Vacuum Magnetic Permeability"))
        {
            _mu_0 = mhd_params.get<double>("Vacuum Magnetic Permeability");
        }

        // Resistivity
        if (mhd_params.isType<bool>("Build Resistive Flux"))
        {
            _build_resistive_flux
                = mhd_params.get<bool>("Build Resistive Flux");
        }

        if (_build_resistive_flux)
        {
            if (mhd_params.isType<bool>("Variable Resistivity"))
            {
                _variable_resistivity
                    = mhd_params.get<bool>("Variable Resistivity");
            }
            if (_variable_resistivity)
            {
                throw std::runtime_error(
                    "No closure models currently exist to evaluate variable "
                    "resistivity. Use a constant resistivity only.");
            }
            else
            {
                _eta = mhd_params.get<double>("Resistivity");
            }
        }

        // Divergence cleaning parameters
        // Hyperbolic divergence cleaning speed
        if (_build_magn_corr)
        {
            _c_h = mhd_params.get<double>(
                "Hyperbolic Divergence Cleaning Speed");
        }
        // Magnetic correction potential equaiton damping factor
        if (mhd_params.isType<double>("Magnetic Correction Damping Factor"))
        {
            _alpha
                = mhd_params.get<double>("Magnetic Correction Damping Factor");
        }
        else
        {
            _alpha = _c_h / 0.18;
        }
    }

    // Build magnetic correction boolean
    KOKKOS_INLINE_FUNCTION bool buildMagnCorr() const
    {
        return _build_magn_corr;
    }

    // Build magnetic correction boolean
    KOKKOS_INLINE_FUNCTION bool buildResistiveFlux() const
    {
        return _build_resistive_flux;
    }

    // Variable resistivity boolean
    KOKKOS_INLINE_FUNCTION bool variableResistivity() const
    {
        return _variable_resistivity;
    }

    // Vacuum magnetic permeability
    KOKKOS_INLINE_FUNCTION double vacuumMagneticPermeability() const
    {
        return _mu_0;
    }

    // Constant resistivity
    KOKKOS_INLINE_FUNCTION double resistivity() const { return _eta; }

    // Divergence cleaning speed
    KOKKOS_INLINE_FUNCTION double hyperbolicDivergenceCleaningSpeed() const
    {
        return _c_h;
    }

    // Magnetic correction damping factor
    KOKKOS_INLINE_FUNCTION double magneticCorrectionDampingFactor() const
    {
        return _alpha;
    }

  private:
    bool _build_magn_corr;
    bool _build_resistive_flux;
    bool _variable_resistivity;
    double _mu_0;
    double _eta;
    double _c_h;
    double _alpha;
};

} // namespace MHDProperties
} // namespace VertexCFD

#endif // VERTEXCFD_FULLINDUCTIONMHDPROPERTIES_HPP

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

using namespace VertexCFD::FluidProperties;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test
{
class IncompressibleConstantFluidPropertiesTest : public ::testing::Test
{
  protected:
    std::unique_ptr<ConstantFluidProperties> cfp;

    virtual void SetUp() override;
    virtual void SetUpDensity();
    virtual void SetUpTemperature();
    virtual void SetUpBuoyancy();
    virtual void SetUpElectricPotential();
};

// Set up all variables but density
void IncompressibleConstantFluidPropertiesTest::SetUp()
{
    Teuchos::ParameterList cfp_params;
    cfp_params.set("Kinematic viscosity", 0.2);
    cfp_params.set("Artificial compressibility", 2.0);
    cfp_params.set("Build Temperature Equation", false);
    cfp = std::make_unique<ConstantFluidProperties>(cfp_params);
}

// Initialize density
void IncompressibleConstantFluidPropertiesTest::SetUpDensity()
{
    Teuchos::ParameterList cfp_params;
    cfp_params.set("Density", 1.5);
    cfp_params.set("Kinematic viscosity", 0.2);
    cfp_params.set("Artificial compressibility", 2.0);
    cfp_params.set("Build Temperature Equation", false);
    cfp = std::make_unique<ConstantFluidProperties>(cfp_params);
}

// With temperature equation
void IncompressibleConstantFluidPropertiesTest::SetUpTemperature()
{
    Teuchos::ParameterList cfp_params;
    cfp_params.set("Kinematic viscosity", 0.2);
    cfp_params.set("Artificial compressibility", 2.0);
    cfp_params.set("Build Temperature Equation", true);
    cfp_params.set("Thermal conductivity", 0.3);
    cfp_params.set("Specific heat capacity", 0.4);
    cfp = std::make_unique<ConstantFluidProperties>(cfp_params);
}

// With buoyancy
void IncompressibleConstantFluidPropertiesTest::SetUpBuoyancy()
{
    Teuchos::ParameterList cfp_params;
    cfp_params.set("Kinematic viscosity", 0.2);
    cfp_params.set("Artificial compressibility", 2.0);
    cfp_params.set("Build Temperature Equation", true);
    cfp_params.set("Build Buoyancy Source", true);
    cfp_params.set("Thermal conductivity", 0.3);
    cfp_params.set("Specific heat capacity", 0.4);
    cfp_params.set("Expansion coefficient", 0.5);
    cfp_params.set("Reference temperature", 0.6);
    cfp = std::make_unique<ConstantFluidProperties>(cfp_params);
}

// With electric potential equation
void IncompressibleConstantFluidPropertiesTest::SetUpElectricPotential()
{
    Teuchos::ParameterList cfp_params;
    cfp_params.set("Kinematic viscosity", 0.2);
    cfp_params.set("Artificial compressibility", 2.0);
    cfp_params.set("Build Temperature Equation", false);
    cfp_params.set("Build Electric Potential Equation", true);
    cfp_params.set("Electrical conductivity", 0.3);
    cfp = std::make_unique<ConstantFluidProperties>(cfp_params);
}

// Density - specified
TEST_F(IncompressibleConstantFluidPropertiesTest, unscaled_density)
{
    IncompressibleConstantFluidPropertiesTest::SetUpDensity();
    const double rho_expect = 1.5;
    const double rho = cfp->constantDensity();
    EXPECT_DOUBLE_EQ(rho_expect, rho);

    const double nu_expect = 0.2;
    const double nu = cfp->constantKinematicViscosity();
    EXPECT_DOUBLE_EQ(nu_expect, nu);

    const double beta_expect = 2.0;
    const double beta = cfp->artificialCompressibility();
    EXPECT_DOUBLE_EQ(beta_expect, beta);
}

// Density - not specified
TEST_F(IncompressibleConstantFluidPropertiesTest, scaled_density)
{
    IncompressibleConstantFluidPropertiesTest::SetUp();
    const double rho_expect = 1.0;
    const double rho = cfp->constantDensity();
    EXPECT_DOUBLE_EQ(rho_expect, rho);

    const double nu_expect = 0.2;
    const double nu = cfp->constantKinematicViscosity();
    EXPECT_DOUBLE_EQ(nu_expect, nu);

    const double beta_expect = 2.0;
    const double beta = cfp->artificialCompressibility();
    EXPECT_DOUBLE_EQ(beta_expect, beta);
}

// Temperature equation
TEST_F(IncompressibleConstantFluidPropertiesTest, temperature_equation)
{
    IncompressibleConstantFluidPropertiesTest::SetUpTemperature();
    const double k_expect = 0.3;
    const double k = cfp->constantThermalConductivity();
    EXPECT_DOUBLE_EQ(k_expect, k);

    const double cp_expect = 0.4;
    const double cp = cfp->constantHeatCapacity();
    EXPECT_DOUBLE_EQ(cp_expect, cp);
}

// Buoyancy
TEST_F(IncompressibleConstantFluidPropertiesTest, bouyancy)
{
    IncompressibleConstantFluidPropertiesTest::SetUpBuoyancy();
    const double beta_expect = 0.5;
    const double beta = cfp->expansionCoefficient();
    EXPECT_DOUBLE_EQ(beta_expect, beta);

    const double T0_expect = 0.6;
    const double T0 = cfp->referenceTemperature();
    EXPECT_DOUBLE_EQ(T0_expect, T0);
}

// Electric potential equation
TEST_F(IncompressibleConstantFluidPropertiesTest, electric_potential_equation)
{
    IncompressibleConstantFluidPropertiesTest::SetUpElectricPotential();
    const double sigma_expect = 0.3;
    const double sigma = cfp->constantElectricalConductivity();
    EXPECT_DOUBLE_EQ(sigma_expect, sigma);
}

//---------------------------------------------------------------------------//

} // end namespace Test

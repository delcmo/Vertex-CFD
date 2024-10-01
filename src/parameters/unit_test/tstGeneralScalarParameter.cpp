#include "VertexCFD_ParameterUnitTestConfig.hpp"

#include "parameters/VertexCFD_GeneralScalarParameter.hpp"

#include <Panzer_Traits.hpp>
#include <Panzer_Workset.hpp>

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <utility>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
double getValue(const panzer::Traits::Residual::ScalarT& observed)
{
    return observed;
}

//---------------------------------------------------------------------------//
double getValue(const panzer::Traits::Jacobian::ScalarT& observed)
{
    return observed.val();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void testGeneralScalarParameter()
{
    // Create input structure.
    std::unordered_map<std::string, std::unordered_map<std::string, double>>
        general_parameter_data;
    {
        std::unordered_map<std::string, double> param_values;
        param_values.emplace("Default Value", 1.2345);
        param_values.emplace("Block 1", 2.3456);
        general_parameter_data.emplace("Foo", std::move(param_values));
    }
    {
        general_parameter_data.emplace(
            "Bar", std::unordered_map<std::string, double>{});
    }

    // Create worksets.
    panzer::Workset w1;
    w1.block_id = "Block 1";
    panzer::Workset w2;
    w2.block_id = "Block 2";

    // Create a parameter.
    typename EvalType::ScalarT result = -1.0;
    Parameter::GeneralScalarParameter<EvalType> gsp("Foo", result);
    EXPECT_EQ("Foo", gsp.name());

    // Test evaluation.
    gsp.update(w1, general_parameter_data);
    EXPECT_EQ(2.3456, getValue(result));
    gsp.update(w2, general_parameter_data);
    EXPECT_EQ(1.2345, getValue(result));

    // Try a parameter not in the list.
    Parameter::GeneralScalarParameter<EvalType> bad_param("Biz", result);
    const std::string biz_msg = "GeneralScalar parameter Biz not found";
    EXPECT_THROW(
        try {
            bad_param.update(w1, general_parameter_data);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(biz_msg, e.what());
            throw;
        },
        std::runtime_error);

    // Try a parameter without a default for a block that doesn't exist.
    Parameter::GeneralScalarParameter<EvalType> no_default("Bar", result);
    const std::string bar_msg
        = "GeneralScalar parameter Bar"
          " does not have a value for block Block 2"
          " and is also missing a default value";
    EXPECT_THROW(
        try {
            no_default.update(w2, general_parameter_data);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(bar_msg, e.what());
            throw;
        },
        std::runtime_error);
}

//---------------------------------------------------------------------------//
TEST(GeneralScalarParameter, residual)
{
    testGeneralScalarParameter<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(GeneralScalarParameter, jacobian)
{
    testGeneralScalarParameter<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD

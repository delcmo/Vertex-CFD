#include "responses/VertexCFD_Response_Utils.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{

//---------------------------------------------------------------------------//
TEST(BuildWorksetDescriptors, FromElementBlocks)
{
    const auto params = Teuchos::ParameterList("Volume Response")
                            .set("Element Blocks", "block1,block2");
    auto workset_descriptors = Response::buildWorksetDescriptors(params);

    ASSERT_EQ(2, workset_descriptors.size());

    EXPECT_EQ("block1", workset_descriptors[0].getElementBlock());
    EXPECT_FALSE(workset_descriptors[0].useSideset());

    EXPECT_EQ("block2", workset_descriptors[1].getElementBlock());
    EXPECT_FALSE(workset_descriptors[1].useSideset());
}

//---------------------------------------------------------------------------//
TEST(BuildWorksetDescriptors, FromSidesets)
{
    const auto params = Teuchos::ParameterList("Surface Response")
                            .set("Sidesets",
                                 Teuchos::ParameterList()
                                     .set("block1", "side1,side2")
                                     .set("block2", "side2"));
    auto workset_descriptors = Response::buildWorksetDescriptors(params);

    ASSERT_EQ(3, workset_descriptors.size());

    EXPECT_EQ("block1", workset_descriptors[0].getElementBlock());
    EXPECT_TRUE(workset_descriptors[0].useSideset());
    EXPECT_EQ("side1", workset_descriptors[0].getSideset());

    EXPECT_EQ("block1", workset_descriptors[1].getElementBlock());
    EXPECT_TRUE(workset_descriptors[1].useSideset());
    EXPECT_EQ("side2", workset_descriptors[1].getSideset());

    EXPECT_EQ("block2", workset_descriptors[2].getElementBlock());
    EXPECT_TRUE(workset_descriptors[2].useSideset());
    EXPECT_EQ("side2", workset_descriptors[2].getSideset());
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD

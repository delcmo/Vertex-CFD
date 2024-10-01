#include "VertexCFD_Utils_VectorizeOutputFieldNames.hpp"

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace VertexCFD;

namespace Test
{
//---------------------------------------------------------------------------//
void vectorizeOutputFieldNamesTest()
{
    Teuchos::ParameterList params;
    params.sublist("Nodal Quantities")
        .set("block_0", "node_val 0,node_val 1,all_val 0")
        .set("block_1", "node_val 1,node_val 2,all_val 1");
    params.sublist("Cell Quantities")
        .set("block_0", "cell_val 0,cell_val 1,all_val 0")
        .set("block_1", "cell_val 1,cell_val 2,all_val 2");
    params.sublist("Cell Average Quantities")
        .set("block_0", "cell_avg_val 0,cell_avg_val 1,all_val 3")
        .set("block_1", "cell_avg_val 1,cell_avg_val 2,all_val 1");
    params.sublist("Cell Average Vectors")
        .set("block_0", "cell_vec_val 0,cell_vec_val 1,all_val 0")
        .set("block_1", "cell_vec_val 1,cell_vec_val 2,all_val 3");

    // Expected returns are vectors of unique field names, which will
    // be ordered
    const std::vector<std::string> exp_fields = {"all_val 0",
                                                 "all_val 1",
                                                 "all_val 2",
                                                 "all_val 3",
                                                 "cell_avg_val 0",
                                                 "cell_avg_val 1",
                                                 "cell_avg_val 2",
                                                 "cell_val 0",
                                                 "cell_val 1",
                                                 "cell_val 2",
                                                 "node_val 0",
                                                 "node_val 1",
                                                 "node_val 2"};
    const std::vector<std::string> exp_vec_fields = {"all_val 0",
                                                     "all_val 3",
                                                     "cell_vec_val 0",
                                                     "cell_vec_val 1",
                                                     "cell_vec_val 2"};

    std::vector<std::string> out_fields;
    std::vector<std::string> out_vector_fields;
    VectorizeOutputFieldNames::getOutputFields(
        params, out_fields, out_vector_fields);

    const int size_ef = exp_fields.size();
    const int size_of = out_fields.size();
    EXPECT_EQ(size_ef, size_of);
    for (int i = 0; i < size_ef; ++i)
    {
        EXPECT_EQ(exp_fields[i], out_fields[i]);
    }

    const int size_evf = exp_vec_fields.size();
    const int size_ovf = out_vector_fields.size();
    EXPECT_EQ(size_evf, size_ovf);
    for (int i = 0; i < size_evf; ++i)
    {
        EXPECT_EQ(exp_vec_fields[i], out_vector_fields[i]);
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(VectorizeOutputFieldNames, test)
{
    vectorizeOutputFieldNamesTest();
}

//---------------------------------------------------------------------------//

} // namespace Test

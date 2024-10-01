#include "VertexCFD_Response_Utils.hpp"

#include <Panzer_String_Utilities.hpp>

namespace VertexCFD
{
namespace Response
{

//---------------------------------------------------------------------------//
std::vector<panzer::WorksetDescriptor>
buildWorksetDescriptors(const Teuchos::ParameterList& plist)
{
    std::vector<panzer::WorksetDescriptor> workset_descriptors;

    // Let user set specific element blocks for volume integrals.
    if (plist.isType<std::string>("Element Blocks"))
    {
        std::vector<std::string> element_blocks;
        panzer::StringTokenizer(
            element_blocks, plist.get<std::string>("Element Blocks"), ",", true);

        workset_descriptors.reserve(element_blocks.size());
        for (const auto& block : element_blocks)
            workset_descriptors.emplace_back(block);
    }
    // Let user set specific sidesets for surface integrals.
    else if (plist.isSublist("Sidesets"))
    {
        const auto& sideset_plist = plist.sublist("Sidesets");
        std::vector<std::string> sidesets;

        for (auto sideset_itr = sideset_plist.begin();
             sideset_itr != sideset_plist.end();
             ++sideset_itr)
        {
            const auto& block = sideset_itr->first;

            sidesets.clear();
            panzer::StringTokenizer(
                sidesets, sideset_plist.get<std::string>(block), ",", true);

            for (const auto& side : sidesets)
                workset_descriptors.emplace_back(block, side);
        }
    }

    return workset_descriptors;
}

} // namespace Response
} // namespace VertexCFD

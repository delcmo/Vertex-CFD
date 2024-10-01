#include <VertexCFD_Utils_Version.hpp>

namespace VertexCFD
{
namespace Utils
{
std::string version()
{
    return VertexCFD_VERSION_STRING;
}

std::string git_commit_hash()
{
    return VertexCFD_GIT_COMMIT_HASH;
}

} // end namespace Utils
} // end namespace VertexCFD

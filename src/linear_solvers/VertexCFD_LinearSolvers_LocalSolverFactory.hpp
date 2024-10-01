#ifndef VERTEXCFD_LINEARSOLVERS_LOCALSOLVERFACTORY_HPP
#define VERTEXCFD_LINEARSOLVERS_LOCALSOLVERFACTORY_HPP

#include "VertexCFD_LinearSolvers_LocalDirectSolver.hpp"

#include <memory>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Class for managing construction of LocalDirectSolver subclasses
//---------------------------------------------------------------------------//
class LocalSolverFactory
{
  public:
    // Prevent construction
    LocalSolverFactory() = delete;

    // Build solver from solver name
    static std::shared_ptr<LocalDirectSolver>
    buildSolver(const Teuchos::ParameterList& params);
};

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD

#endif // VERTEXCFD_LINEARSOLVERS_LOCALSOLVERFACTORY_HPP

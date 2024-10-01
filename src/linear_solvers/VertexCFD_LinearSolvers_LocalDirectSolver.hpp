#ifndef VERTEXCFD_LINEARSOLVERS_LOCALDIRECTSOLVER_HPP
#define VERTEXCFD_LINEARSOLVERS_LOCALDIRECTSOLVER_HPP

#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_RowMatrix.hpp>

#include <vector>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Base class for solvers performing local (non-MPI) sparse direct solves
//---------------------------------------------------------------------------//
class LocalDirectSolver
{
  public:
    // Constructor
    LocalDirectSolver() = default;

    // Set or change local matrix
    virtual void setMatrix(Teuchos::RCP<const Tpetra::RowMatrix<>> A) = 0;

    // Perform one-time initialization (e.g., symbolic factorization)
    virtual void initialize() = 0;

    // Compute factorization (every time matrix changes)
    virtual void compute() = 0;

    // Given RHS vector b, solve for x
    virtual void solve(const Tpetra::MultiVector<>& b, Tpetra::MultiVector<>& x)
        = 0;
};

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD

#endif // VERTEXCFD_LINEARSOLVERS_LOCALDIRECTSOLVER_HPP

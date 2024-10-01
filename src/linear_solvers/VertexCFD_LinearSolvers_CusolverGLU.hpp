#ifndef VERTEXCFD_LINEARSOLVERS_CUSOLVERGLU_HPP
#define VERTEXCFD_LINEARSOLVERS_CUSOLVERGLU_HPP

#include "VertexCFD_LinearSolvers_CusolverNonpublic.hpp"
#include "VertexCFD_LinearSolvers_LocalDirectSolver.hpp"

#include <Teuchos_RCP.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_RowMatrix.hpp>

#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <thrust/device_vector.h>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Local preconditioner/solver using cuSOLVER GLU for on-GPU solves.
// This class uses the nonpublic GLU solver from the NVIDIA cuSOLVER library.
// This is a refactor-based solver where an initial factorization is performed
// on the host (using a corresponding cuSOLVER routine) and subsequent
// factorizations are performed on the GPU. The initial host-side
// factorization can be reused across multiple nonlinear iterations within a
// time step as well as multiple time steps. This process assumes that
// 1) the sparsity pattern of the matrix does not change, and 2) the locations
// of any necessary pivoting that are determined by the host-side
// factorization remain valid for all subsequent GPU solves. Effectively,
// the GPU solves are utilizing static pivoting. The solver does not currently
// attempt to recover if a factorization breaks down due to pivots becoming
// stale/invalid.
//---------------------------------------------------------------------------//
class CusolverGLU : public LocalDirectSolver
{
  private:
    // >>> DATA

    // Matrix
    Teuchos::RCP<const Tpetra::RowMatrix<>> _A;

    // Device-side matrix data
    thrust::device_vector<int> _A_rowptr;
    thrust::device_vector<int> _A_colind;
    thrust::device_vector<double> _A_values;

    // Host-side matrix data
    std::vector<int> _A_rowptr_host;
    std::vector<int> _A_colind_host;
    std::vector<double> _A_values_host;

    // Pivoting data
    double _pivot_threshold;
    int _reorder;

    // Scratch space
    thrust::device_vector<char> _work;

    // Persistent Cusparse info
    cusolverSpHandle_t _handle;
    csrluInfoHost_t _lu_info;
    csrgluInfo_t _M_info;
    cusparseMatDescr_t _A_descr, _M_descr;

    // Status flags
    bool _matrix_set;
    bool _initialized;
    bool _computed;

  public:
    // Constructor
    CusolverGLU(const Teuchos::ParameterList& params);

    // Update internal matrix
    void setMatrix(Teuchos::RCP<const Tpetra::RowMatrix<>> A) override;

    // Inherited interface from LocalDirectSolver
    void initialize() override;
    void compute() override;

    // Inherited interface from LocalDirectSolver
    void
    solve(const Tpetra::MultiVector<>& b, Tpetra::MultiVector<>& x) override;

  private:
    // Check status condition and throw exception if failed
    void
    check_status(cusolverStatus_t stat, const std::string& identifier) const;

    // Get number of rows in matrix
    int num_local_rows() const;
    std::size_t num_local_entries() const;
    std::size_t max_entries_per_row() const;
};

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD

#endif // VERTEXCFD_LINEARSOLVERS_CUSOLVERGLU_HPP

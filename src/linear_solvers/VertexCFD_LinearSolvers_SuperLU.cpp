#include "VertexCFD_LinearSolvers_SuperLU.hpp"

#include <Tpetra_Vector.hpp>

#include <cassert>
#include <sstream>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Constructor
//---------------------------------------------------------------------------//
SuperLUSolver::SuperLUSolver()
    : _matrix_set(false)
    , _initialized(false)
    , _computed(false)
    , _factored(false)
{
    // setup grid 1 x 1 x 1 for single task MPI solve
    superlu_gridinit(MPI_COMM_SELF, 1, 1, &_grid);

    // set SuperLU solver options
    set_default_options_dist(&_options);
    //_options.Algo3d = YES;
    // 0: turn off row permutation
    _options.RowPerm = (rowperm_t)0;
    // 4:  Use METIS ordering on A'+A
    _options.ColPerm = (colperm_t)4;

    _options.IterRefine = NOREFINE;
    _options.PrintStat = NO;

    // print out SuperLU version
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (comm_rank == 0)
    {
        int v_major, v_minor, v_bugfix;
        superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
        std::cout << "Uses SuperLU version " << v_major << "." << v_minor
                  << "." << v_bugfix << std::endl;
        print_options_dist(&_options);
        fflush(stdout);
    }
}

SuperLUSolver::~SuperLUSolver()
{
    Destroy_CompRowLoc_Matrix_dist(&_A);
    dDestroy_LU(_num_rows, &_grid, &_LUstruct);

    dScalePermstructFree(&_ScalePermstruct);
    dLUstructFree(&_LUstruct);
    if (_options.SolveInitialized)
    {
        dSolveFinalize(&_options, &_SOLVEstruct);
    }

    superlu_gridexit(&_grid);
}

//---------------------------------------------------------------------------//
// Set matrix
//---------------------------------------------------------------------------//
void SuperLUSolver::setMatrix(Teuchos::RCP<const Tpetra::RowMatrix<>> A)
{
    if (_matrix_set)
    {
        // WARNING:
        // destroying _A also destroys the vectors _A_rowptr_host,
        // _A_colind_host and _A_values_host!
        Destroy_CompRowLoc_Matrix_dist(&_A);
        dDestroy_LU(_num_rows, &_grid, &_LUstruct);
    }

    _num_rows = A->getNodeNumRows();

    // Host-side allocation for CRS data
    // Since SuperLU will free these arrays when destroying matrix using its
    // own wrappers, let's use SuperLU wrappers to allocate that memory to
    // ensure consistency
    int nnz_loc = A->getNodeNumEntries();
    _A_rowptr_host = (int*)SUPERLU_MALLOC((_num_rows + 1) * sizeof(int));
    _A_colind_host = (int*)SUPERLU_MALLOC(nnz_loc * sizeof(int));
    _A_values_host = (double*)SUPERLU_MALLOC(nnz_loc * sizeof(double));

    // Extract matrix data from A
    std::size_t num_entries;
    Teuchos::Array<int> row_inds;
    Teuchos::Array<double> row_vals;
    _A_rowptr_host[0] = 0;
    int offset = 0;
    for (int row = 0; row < _num_rows; ++row)
    {
        num_entries = A->getNumEntriesInLocalRow(row);
        row_inds.resize(num_entries);
        row_vals.resize(num_entries);
        A->getLocalRowCopy(row, row_inds, row_vals, num_entries);
        memcpy(_A_colind_host + offset,
               row_inds.data(),
               num_entries * sizeof(int));
        memcpy(_A_values_host + offset,
               row_vals.data(),
               num_entries * sizeof(double));
        _A_rowptr_host[1 + row] = _A_rowptr_host[row] + num_entries;
        offset += num_entries;
    }

    // create SuperLU matrix
    dCreate_CompRowLoc_Matrix_dist(&_A,
                                   _num_rows,
                                   _num_rows,
                                   nnz_loc,
                                   _num_rows,
                                   0,
                                   _A_values_host,
                                   _A_colind_host,
                                   _A_rowptr_host,
                                   SLU_NR_loc,
                                   SLU_D,
                                   SLU_GE);

    if (_factored)
        _options.Fact = SamePattern; // tell SuperLU new matrix has same nnz
                                     // pattern
    else
        _options.Fact = DOFACT;

    _matrix_set = true;
    _factored = false;
}

//---------------------------------------------------------------------------//
// Initialize preconditioner
//---------------------------------------------------------------------------//
void SuperLUSolver::initialize()
{
    assert(_matrix_set);

    if (_initialized)
    {
        return;
    }

    // Initialize SuperLU structs: ScalePermstruct and LUstruct
    // Needs to be done only once since saprsity pattern will not change
    dScalePermstructInit(_num_rows, _num_rows, &_ScalePermstruct);
    dLUstructInit(_num_rows, &_LUstruct);

    _initialized = true;
}

//---------------------------------------------------------------------------//
// Compute preconditioner
//---------------------------------------------------------------------------//
void SuperLUSolver::compute()
{
    _computed = true;
}

//---------------------------------------------------------------------------//
// Compute and apply preconditioner
//---------------------------------------------------------------------------//
void SuperLUSolver::solve(const Tpetra::MultiVector<>& b,
                          Tpetra::MultiVector<>& x)
{
    assert(_matrix_set);
    assert(_initialized);

    Kokkos::fence();

    // copy b into x
    Tpetra::deep_copy(x, b);

    // Get Kokkos Views
    x.clear_sync_state();
    auto b_view = b.getLocalViewHost(Tpetra::Access::ReadOnly);

    auto xvec = x.getVectorNonConst(0);
    auto x_view = xvec->getLocalViewHost(Tpetra::Access::OverwriteAllStruct());

    int info;

    // Initialize SuperLU statistics variables
    SuperLUStat_t stat;
    PStatInit(&stat);

    double berr;

    // factorize and solve linear system
    // by using Gaussian elimination with static pivoting
    // Output:
    // xv is overwritten with the solution
    // A is overwritten by the scaled and permuted
    // matrix diag(R)*A*diag(C)*Pc^T
    // Solver expects and returns data on the CPU
    // Detailed info in SuperLU_dist file: SRC/pdgssvx.c
    double* xv = x_view.data();

    if (_factored)
        _options.Fact = FACTORED; // factored form of A is supplied

    pdgssvx(&_options,
            &_A,
            &_ScalePermstruct,
            xv,
            _num_rows,
            1,
            &_grid,
            &_LUstruct,
            &_SOLVEstruct,
            &berr,
            &stat,
            &info);

    x.modify_host();
    x.sync_device();
    if (info)
    { // Something is wrong
        std::stringstream ss;
        ss << "ERROR: INFO = " << info << " returned from pdgssvx3d()";
        throw std::runtime_error(ss.str());
    }

    // Print statistics
    // PStatPrint (&_options, &stat, &_grid);

    PStatFree(&stat);

    _factored = true;
}

} // namespace LinearSolvers
} // namespace VertexCFD

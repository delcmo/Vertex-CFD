#ifndef VERTEXCFD_LINEARSOLVERS_SUPERLUSOLVER_HPP
#define VERTEXCFD_LINEARSOLVERS_SUPERLUSOLVER_HPP

#include "VertexCFD_LinearSolvers_LocalDirectSolver.hpp"

// superlu_ddefs.h includes C macro that conflicts with some
// cuda libraries include files. So let's include it after
// everything else
#define GPU_ACC
#include <superlu_ddefs.h>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Local preconditioner/solver using SuperLU for on-GPU solves.
//---------------------------------------------------------------------------//
class SuperLUSolver : public LocalDirectSolver
{
    static_assert(std::is_same<int_t, int>::value,
                  "SuperLU should be build with int_t=int");

  private:
    // >>> DATA

    // Matrix
    SuperMatrix _A;

    // Host-side matrix data
    int* _A_rowptr_host;
    int* _A_colind_host;
    double* _A_values_host;

    // Persistent SuperLU info
    gridinfo_t _grid;
    superlu_dist_options_t _options;
    dLUstruct_t _LUstruct;
    dScalePermstruct_t _ScalePermstruct;
    dSOLVEstruct_t _SOLVEstruct;

    // Status flags
    bool _matrix_set;
    bool _initialized;
    bool _computed;
    bool _factored;

    int _num_rows;

  public:
    // Constructor
    SuperLUSolver();

    ~SuperLUSolver();

    // Update internal matrix
    void setMatrix(Teuchos::RCP<const Tpetra::RowMatrix<>> A) override;

    // Inherited interface from LocalDirectSolver
    void initialize() override;
    void compute() override;

    // Inherited interface from LocalDirectSolver
    void
    solve(const Tpetra::MultiVector<>& b, Tpetra::MultiVector<>& x) override;
};

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD

#endif // VERTEXCFD_LINEARSOLVERS_SUPERLUSOLVER_HPP

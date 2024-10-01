#ifndef VERTEXCFD_LINEARSOLVERS_PRECONDITIONER_HPP
#define VERTEXCFD_LINEARSOLVERS_PRECONDITIONER_HPP

#include "VertexCFD_LinearSolvers_LocalDirectSolver.hpp"

#include <Ifpack2_Details_CanChangeMatrix.hpp>
#include <Ifpack2_Preconditioner.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_RowMatrix.hpp>

#include <memory>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
/*
 * This class is the Trilinos (Ifpack2) interface for local domain
 * preconditioners as part of an additive Schwarz type scheme. It is assumed
 * that the matrix provided via the setMatrix method will only contain the
 * _local_ matrix elements with no off-processor components.
 * It implements the Ifpack2::Preconditioner interface, which allows it to
 * be used as the local solver within an Ifpack2::AdditiveSchwarz instance.
 * This class is only an interface -- the local solve itself is handled by
 * a subclass of LocalDirectSolver and built by the LocalSolverFactory. This
 * design prevents every local solver class (e.g., CuSOLVER, SuperLU) from
 * having to re-implement the full API inherited from Ifpack2::Preconditioner.
 */
//---------------------------------------------------------------------------//
class Preconditioner
    : public Ifpack2::Preconditioner<>,
      public Ifpack2::Details::CanChangeMatrix<Tpetra::RowMatrix<>>
{
  private:
    using MV = Tpetra::MultiVector<>;

    // Local matrix
    Teuchos::RCP<const Tpetra::RowMatrix<>> _A;

    // Implementation of local solve
    std::shared_ptr<LocalDirectSolver> _local_solver;

    // Tracking of calls and time to init, compute, apply
    bool _initialized;
    bool _computed;
    int _num_initialize;
    int _num_compute;
    mutable int _num_apply; // Must be updated from within const "apply"
    double _init_time;
    double _compute_time;
    mutable double _apply_time;

    // Timer labels
    const std::string _set_label = "VertexCFD::Preconditioner::setMatrix";
    const std::string _init_label = "VertexCFD::Preconditioner::initialize";
    const std::string _compute_label = "VertexCFD::Preconditioner::compute";
    const std::string _apply_label = "VertexCFD::Preconditioner::apply";

  public:
    // Constructor
    Preconditioner();

    // Inherited API from Ifpack2::CanChangeMatrix
    void setMatrix(const Teuchos::RCP<const Tpetra::RowMatrix<>>& A) override;

    // Inherited API from Ifpack2::Preconditioner
    Teuchos::RCP<const Tpetra::RowMatrix<>> getMatrix() const override
    {
        return _A;
    }
    bool isInitialized() const override { return _initialized; }
    bool isComputed() const override { return _computed; }
    int getNumInitialize() const override { return _num_initialize; }
    int getNumCompute() const override { return _num_compute; }
    int getNumApply() const override { return _num_apply; }
    double getInitializeTime() const override { return _init_time; }
    double getComputeTime() const override { return _compute_time; }
    double getApplyTime() const override { return _apply_time; }
    Teuchos::RCP<const Tpetra::Map<>> getDomainMap() const override
    {
        return _A->getDomainMap();
    }
    Teuchos::RCP<const Tpetra::Map<>> getRangeMap() const override
    {
        return _A->getRangeMap();
    }
    void setParameters(const Teuchos::ParameterList& pl) override;
    void initialize() override;
    void compute() override;
    void apply(const Tpetra::MultiVector<>& x,
               Tpetra::MultiVector<>& y,
               Teuchos::ETransp mode,
               double alpha,
               double beta) const override;
};

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD

#endif // VERTEXCFD_LINEARSOLVERS_PRECONDITIONER_HPP

#ifndef VERTEXCFD_TEMPUSOBSERVER_WRITEMATRIX_HPP
#define VERTEXCFD_TEMPUSOBSERVER_WRITEMATRIX_HPP

#include <Tempus_Integrator.hpp>
#include <Tempus_IntegratorObserver.hpp>

#include <Panzer_GlobalIndexer.hpp>
#include <Teuchos_ParameterList.hpp>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
class WriteMatrix : virtual public Tempus::IntegratorObserver<Scalar>
{
  public:
    WriteMatrix(const Teuchos::ParameterList& output_params);

    /// Observe the beginning of the time integrator.
    void observeStartIntegrator(
        const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe the beginning of the time step loop.
    void
    observeStartTimeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe after the next time step size is selected. The
    /// observer can choose to change the current integratorStatus.
    void
    observeNextTimeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe before Stepper takes step.
    void
    observeBeforeTakeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe after Stepper takes step.
    void
    observeAfterTakeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe after checking time step. Observer can still fail the time step
    /// here.
    void observeAfterCheckTimeStep(
        const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe the end of the time step loop.
    void
    observeEndTimeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe the end of the time integrator.
    void
    observeEndIntegrator(const Tempus::Integrator<Scalar>& integrator) override;

  private:
    Teuchos::Array<int> _write_steps;
    std::string _jacobian_prefix;
    std::string _residual_prefix;
    bool _write_residual;

    // Extract Thyra linear operator from linear solver
    Teuchos::RCP<const Thyra::LinearOpBase<Scalar>> extractLinearOp(
        const Teuchos::RCP<const Thyra::LinearOpWithSolveBase<Scalar>>&
            linear_solver) const;

    // Extract forward op src (the Jacobian) from a linear op with solve
    template<class SolverType>
    Teuchos::RCP<const Thyra::LinearOpBase<Scalar>> tryExtractForwardOp(
        const Teuchos::RCP<const Thyra::LinearOpWithSolveBase<Scalar>>&
            linear_solver) const;

    // Write Jacobian matrix to file, toggling on Epetra vs. Tpetra
    void
    write_jacobian(const Teuchos::RCP<const Thyra::LinearOpBase<Scalar>>& matrix,
                   const std::string& filename,
                   const std::string& description) const;

    // Write residual to file, toggling on Epetra vs. Tpetra
    void
    write_residual(const Teuchos::RCP<const Thyra::VectorBase<Scalar>>& vec,
                   const std::string& filename,
                   const std::string& description) const;
};

//---------------------------------------------------------------------------//

} // end namespace TempusObserver
} // end namespace VertexCFD

#include "VertexCFD_TempusObserver_WriteMatrix_impl.hpp"

#endif // end VERTEXCFD_TEMPUSOBSERVER_WRITEMATRIX_HPP

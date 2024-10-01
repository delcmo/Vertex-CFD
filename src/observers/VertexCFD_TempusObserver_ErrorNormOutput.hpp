#ifndef VERTEXCFD_TEMPUSOBSERVER_ERRORNORMOUTPUT_HPP
#define VERTEXCFD_TEMPUSOBSERVER_ERRORNORMOUTPUT_HPP

#include <observers/VertexCFD_Compute_ErrorNorms.hpp>

#include <Tempus_Integrator.hpp>
#include <Tempus_IntegratorObserver.hpp>

#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <limits>
#include <string>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
class ErrorNormOutput : virtual public Tempus::IntegratorObserver<Scalar>
{
  public:
    ErrorNormOutput(
        const Teuchos::ParameterList& error_norm_list,
        Teuchos::RCP<ComputeErrorNorms::ErrorNorms<Scalar>> error_norm);

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
    /// Print error norms with heading
    void
    print_error_norms(const std::string& heading,
                      const std::vector<typename ComputeErrorNorms::ErrorNorms<
                          Scalar>::DofErrorNorm>& error_norm,
                      const double scaling = 1.0);

    int _output_freq;
    bool _compute_time_error;
    Teuchos::FancyOStream _ostream;
    Teuchos::RCP<ComputeErrorNorms::ErrorNorms<Scalar>> _error_norm;
    static constexpr int prec = std::numeric_limits<double>::digits10 + 1;
    std::vector<typename ComputeErrorNorms::ErrorNorms<Scalar>::DofErrorNorm>
        _L1_time_error_norms;
    std::vector<typename ComputeErrorNorms::ErrorNorms<Scalar>::DofErrorNorm>
        _L2_time_error_norms;
};

//---------------------------------------------------------------------------//

} // end namespace TempusObserver
} // end namespace VertexCFD

#include "VertexCFD_TempusObserver_ErrorNormOutput_impl.hpp"

#endif // end VERTEXCFD_TEMPUSOBSERVER_ERRORNORMOUTPUT_HPP

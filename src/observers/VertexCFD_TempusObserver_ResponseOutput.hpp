#ifndef VERTEXCFD_TEMPUSOBSERVER_RESPONSEOUTPUT_HPP
#define VERTEXCFD_TEMPUSOBSERVER_RESPONSEOUTPUT_HPP

#include "responses/VertexCFD_ResponseManager.hpp"

#include <Tempus_Integrator.hpp>
#include <Tempus_IntegratorObserver.hpp>

#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCP.hpp>

#include <vector>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
class ResponseOutput : virtual public Tempus::IntegratorObserver<Scalar>
{
  public:
    ResponseOutput(Teuchos::RCP<Response::ResponseManager> response_manager,
                   std::vector<int> output_freq);

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
    Teuchos::FancyOStream _ostream;
    Teuchos::RCP<Response::ResponseManager> _response_manager;
    std::vector<int> _output_freq;

    void outputResponses(const Tempus::Integrator<Scalar>& integrator,
                         const int current_index = 0);
};

//---------------------------------------------------------------------------//

} // namespace TempusObserver
} // namespace VertexCFD

#include "VertexCFD_TempusObserver_ResponseOutput_impl.hpp"

#endif // VERTEXCFD_TEMPUSOBSERVER_RESPONSEOUTPUT_HPP

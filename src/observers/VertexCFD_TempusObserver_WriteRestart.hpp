#ifndef VERTEXCFD_TEMPUSOBSERVER_WRITERESTART_HPP
#define VERTEXCFD_TEMPUSOBSERVER_WRITERESTART_HPP

#include "mesh/VertexCFD_Mesh_Restart.hpp"

#include <Tempus_Integrator.hpp>
#include <Tempus_IntegratorObserver.hpp>

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_ResponseLibrary.hpp>
#include <Panzer_STK_Interface.hpp>
#include <Panzer_STK_ResponseEvaluatorFactory_SolutionWriter.hpp>
#include <Panzer_STK_Utilities.hpp>

#include <Teuchos_ParameterList.hpp>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
class WriteRestart : virtual public Tempus::IntegratorObserver<Scalar>
{
  public:
    WriteRestart(const Teuchos::RCP<Mesh::RestartWriter>& restart_writer,
                 const Teuchos::ParameterList& output_params);

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
    void writeSolution(const Tempus::Integrator<Scalar>& integrator);

  private:
    Teuchos::RCP<Mesh::RestartWriter> _restart_writer;
    int _write_frequency = 1;
    int _last_index = -1;
};

//---------------------------------------------------------------------------//

} // end namespace TempusObserver
} // end namespace VertexCFD

#include "VertexCFD_TempusObserver_WriteRestart_impl.hpp"

#endif // end VERTEXCFD_TEMPUSOBSERVER_WRITERESTART_HPP

#ifndef VERTEXCFD_TEMPUSOBSERVER_WRITERESTART_IMPL_HPP
#define VERTEXCFD_TEMPUSOBSERVER_WRITERESTART_IMPL_HPP

#include <string>
#include <vector>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
WriteRestart<Scalar>::WriteRestart(
    const Teuchos::RCP<Mesh::RestartWriter>& restart_writer,
    const Teuchos::ParameterList& output_params)
    : _restart_writer(restart_writer)
{
    if (output_params.isType<int>("Restart Write Frequency"))
    {
        _write_frequency
            = output_params.template get<int>("Restart Write Frequency");
    }
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteRestart<Scalar>::observeStartIntegrator(
    const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteRestart<Scalar>::observeStartTimeStep(const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteRestart<Scalar>::observeNextTimeStep(const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteRestart<Scalar>::observeBeforeTakeStep(
    const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteRestart<Scalar>::observeAfterTakeStep(const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteRestart<Scalar>::observeAfterCheckTimeStep(
    const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteRestart<Scalar>::observeEndTimeStep(
    const Tempus::Integrator<Scalar>& integrator)
{
    // Only write solution at specified time step intervals.
    if (0 == integrator.getIndex() % _write_frequency)
    {
        writeSolution(integrator);
    }
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteRestart<Scalar>::observeEndIntegrator(
    const Tempus::Integrator<Scalar>& integrator)
{
    // Write out final solution, but only if this time step has not been
    // written already.
    if (integrator.getIndex() > _last_index)
    {
        writeSolution(integrator);
    }
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteRestart<Scalar>::writeSolution(
    const Tempus::Integrator<Scalar>& integrator)
{
    auto time = integrator.getTime();
    auto solution = integrator.getSolutionHistory()->findState(time)->getX();
    auto solution_dot
        = integrator.getSolutionHistory()->findState(time)->getXDot();
    _last_index = integrator.getIndex();
    _restart_writer->writeSolution(solution, solution_dot, _last_index, time);
}

//---------------------------------------------------------------------------//

} // end namespace TempusObserver
} // end namespace VertexCFD

#endif // end VERTEXCFD_TEMPUSOBSERVER_WRITERESTART_IMPL_HPP

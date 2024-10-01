#ifndef VERTEXCFD_TEMPUSOBSERVER_WRITETOEXODUS_IMPL_HPP
#define VERTEXCFD_TEMPUSOBSERVER_WRITETOEXODUS_IMPL_HPP

#include <string>
#include <vector>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
WriteToExodus<Scalar>::WriteToExodus(
    const Teuchos::RCP<Mesh::ExodusWriter>& exodus_writer,
    const Teuchos::ParameterList& output_params)
    : _exodus_writer(exodus_writer)
    , _write_frequency(1)
{
    _write_frequency
        = output_params.template get<int>("Exodus Write Frequency");
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteToExodus<Scalar>::observeStartIntegrator(
    const Tempus::Integrator<Scalar>& integrator)
{
    // Write out initial conditions.
    writeSolution(integrator);
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteToExodus<Scalar>::observeStartTimeStep(
    const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteToExodus<Scalar>::observeNextTimeStep(const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteToExodus<Scalar>::observeBeforeTakeStep(
    const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteToExodus<Scalar>::observeAfterTakeStep(
    const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteToExodus<Scalar>::observeAfterCheckTimeStep(
    const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteToExodus<Scalar>::observeEndTimeStep(
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
void WriteToExodus<Scalar>::observeEndIntegrator(
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
void WriteToExodus<Scalar>::writeSolution(
    const Tempus::Integrator<Scalar>& integrator)
{
    const auto state = integrator.getSolutionHistory()->getCurrentState();
    const auto time = state->getTime();
    const auto time_step = state->getTimeStep();
    const auto solution = state->getX();
    const auto solution_dot = state->getXDot();
    _exodus_writer->writeSolution(solution, solution_dot, time, time_step);
    _last_index = integrator.getIndex();
}

//---------------------------------------------------------------------------//

} // end namespace TempusObserver
} // end namespace VertexCFD

#endif // end VERTEXCFD_TEMPUSOBSERVER_WRITETOEXODUS_IMPL_HPP

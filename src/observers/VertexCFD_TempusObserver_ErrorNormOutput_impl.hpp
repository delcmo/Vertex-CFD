#ifndef VERTEXCFD_TEMPUSOBSERVER_ERRORNORMOUTPUT_IMPL_HPP
#define VERTEXCFD_TEMPUSOBSERVER_ERRORNORMOUTPUT_IMPL_HPP

#include <iomanip>
#include <iostream>
#include <limits>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
ErrorNormOutput<Scalar>::ErrorNormOutput(
    const Teuchos::ParameterList& error_norm_list,
    Teuchos::RCP<ComputeErrorNorms::ErrorNorms<Scalar>> error_norm)
    : _output_freq(std::numeric_limits<int>::max())
    , _compute_time_error(true)
    , _ostream(Teuchos::rcp(&std::cout, false))
    , _error_norm(error_norm)
{
    // Get output frequency
    if (error_norm_list.isType<int>("Output Frequency"))
    {
        _output_freq = error_norm_list.get<int>("Output Frequency");
    }

    // Time integrated error norm
    if (error_norm_list.isType<bool>("Compute Time Integral"))
    {
        _compute_time_error
            = error_norm_list.get<bool>("Compute Time Integral");
    }

    // Initialize L1/L2 error norms objects
    if (_compute_time_error)
    {
        _L1_time_error_norms = _error_norm->L1_errorNorms();
        _L2_time_error_norms = _error_norm->L2_errorNorms();
    }

    // Initialize output variable
    _ostream.setShowProcRank(false);
    _ostream.setOutputToRootOnly(0);
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNormOutput<Scalar>::observeStartIntegrator(
    const Tempus::Integrator<Scalar>& integrator)
{
    // Compute Initial Error Norms
    const auto state = integrator.getSolutionHistory()->getCurrentState();
    _error_norm->ComputeNorms(state);

    // Print initial L1/L2 error norms
    print_error_norms("Initial Spatial Integrated L1 Error Norms:",
                      _error_norm->L1_errorNorms());
    print_error_norms("Initial Spatial Integrated L2 Error Norms:",
                      _error_norm->L2_errorNorms());
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNormOutput<Scalar>::observeStartTimeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNormOutput<Scalar>::observeNextTimeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNormOutput<Scalar>::observeBeforeTakeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNormOutput<Scalar>::observeAfterTakeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNormOutput<Scalar>::observeAfterCheckTimeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNormOutput<Scalar>::observeEndTimeStep(
    const Tempus::Integrator<Scalar>& integrator)
{
    // Compute Error Norms
    const auto state = integrator.getSolutionHistory()->getCurrentState();
    _error_norm->ComputeNorms(state);
    const auto dt = state->getTimeStep();
    const auto current_index = state->getIndex();

    // Compute integrated L1/L2 error norms
    if (_compute_time_error)
    {
        for (std::size_t i = 0; i < _L1_time_error_norms.size(); ++i)
        {
            _L1_time_error_norms[i].error_norm
                += _error_norm->L1_errorNorms()[i].error_norm * dt;
        }

        for (std::size_t i = 0; i < _L2_time_error_norms.size(); ++i)
        {
            _L2_time_error_norms[i].error_norm
                += _error_norm->L2_errorNorms()[i].error_norm * dt;
        }
    }

    // Print L1/L2 error norms
    if (0 == current_index % _output_freq)
    {
        print_error_norms("Spatial Integrated L1 Error Norms:",
                          _error_norm->L1_errorNorms());
        print_error_norms("Spatial Integrated L2 Error Norms:",
                          _error_norm->L2_errorNorms());

        if (_compute_time_error)
        {
            const double current_time = integrator.getTime();

            print_error_norms("Temporal/Spatial Integrated L1 Error Norms:",
                              _L1_time_error_norms,
                              current_time);

            print_error_norms("Temporal/Spatial Integrated L2 Error Norms:",
                              _L2_time_error_norms,
                              current_time);
        }
    }
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNormOutput<Scalar>::observeEndIntegrator(
    const Tempus::Integrator<Scalar>& integrator)
{
    if (_compute_time_error)
    {
        // Get final time
        const double final_time = integrator.getTime();

        // L1 error norms
        print_error_norms("Final Temporal/Spatial Integrated L1 Error Norms:",
                          _L1_time_error_norms,
                          final_time);

        // L2 error norms
        print_error_norms("Final Temporal/Spatial Integrated L2 Error Norms:",
                          _L2_time_error_norms,
                          final_time);
    }
    else
    {
        // Print L1/L2 error norms
        print_error_norms("Final Spatial Integrated L1 Error Norms:",
                          _error_norm->L1_errorNorms());
        print_error_norms("Final Spatial Integrated L2 Error Norms:",
                          _error_norm->L2_errorNorms());
    }
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNormOutput<Scalar>::print_error_norms(
    const std::string& heading,
    const std::vector<
        typename ComputeErrorNorms::ErrorNorms<Scalar>::DofErrorNorm>& error_norm,
    const double scaling)
{
    _ostream << '\n'
             << heading << '\n'
             << std::setprecision(prec) << std::scientific;
    for (auto& dof : error_norm)
    {
        _ostream << "   " << dof.name << " = " << dof.error_norm / scaling
                 << '\n';
    }
    _ostream << '\n';
}

//---------------------------------------------------------------------------//

} // end namespace TempusObserver
} // end namespace VertexCFD

#endif // end VERTEXCFD_TEMPUSOBSERVER_ERRORNORMOUTPUT_IMPL_HPP

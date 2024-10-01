#ifndef VERTEXCFD_NOXOBSERVER_ITERATIONOUTPUT_HPP
#define VERTEXCFD_NOXOBSERVER_ITERATIONOUTPUT_HPP

#include <NOX.H>
#include <NOX_Observer.hpp>
#include <NOX_Thyra_Group.H>

#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace NOXObserver
{
//---------------------------------------------------------------------------//
class IterationOutput : public NOX::Observer
{
  public:
    IterationOutput();

    //! User defined method that will be executed at the start of a call to
    //! NOX::Solver::Generic::step().
    void runPreIterate(const NOX::Solver::Generic& solver) override;

    //! User defined method that will be executed at the end of a call to
    //! NOX::Solver::Generic::step().
    void runPostIterate(const NOX::Solver::Generic& solver) override;

    //! User defined method that will be executed at the start of a call to
    //! NOX::Solver::Generic::solve().
    void runPreSolve(const NOX::Solver::Generic& solver) override;

    //! User defined method that will be executed at the end of a call to
    //! NOX::Solver::Generic::solve().
    void runPostSolve(const NOX::Solver::Generic& solver) override;

  private:
    Teuchos::FancyOStream _ostream;
};

//---------------------------------------------------------------------------//
// This class is a workaround for the unused lineaer solver status in
// NOX::SolverStats::LinearSolveStats::LogLinearSolve()
class SolveDataExtractor : public NOX::Thyra::Group
{
  public:
    SolveDataExtractor(const NOX::Thyra::Group& group);
    int lastLinearSolveNumIters() const;
    double lastLinearSolveAchievedTol() const;
};

//---------------------------------------------------------------------------//

} // end namespace NOXObserver
} // end namespace VertexCFD

#endif // end VERTEXCFD_NOXOBSERVER_ITERATIONOUTPUT_HPP

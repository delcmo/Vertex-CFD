#ifndef VERTEXCFD_TEMPUSTIMESTEPCONTROL_STRATEGY_HPP
#define VERTEXCFD_TEMPUSTIMESTEPCONTROL_STRATEGY_HPP

#include <Tempus_TimeStepControlStrategy.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace TempusTimeStepControl
{
//---------------------------------------------------------------------------//
template<class Scalar>
class Strategy : virtual public Tempus::TimeStepControlStrategy<Scalar>
{
  public:
    virtual ~Strategy() = default;
    double currentCFL() const { return _cfl_current; }

  protected:
    void setCurrentCFL(const double cfl) { _cfl_current = cfl; }

  private:
    double _cfl_current = 0.0;
};

//---------------------------------------------------------------------------//

} // namespace TempusTimeStepControl
} // namespace VertexCFD

#endif // VERTEXCFD_TEMPUSTIMESTEPCONTROL_STRATEGY_HPP

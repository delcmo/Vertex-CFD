#ifndef VERTEXCFD_MESH_RESTART_HPP
#define VERTEXCFD_MESH_RESTART_HPP

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_STK_Interface.hpp>

#include <Thyra_VectorBase.hpp>

#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace VertexCFD
{
namespace Mesh
{
//---------------------------------------------------------------------------//
class Restart
{
  public:
    ~Restart() = default;

    MPI_Datatype
    setupDofMapData(const Teuchos::RCP<const panzer_stk::STK_Interface>& mesh,
                    const Teuchos::RCP<const panzer::GlobalIndexer>& dof_manager,
                    int& dofmap_offset,
                    int& local_num_own_elem,
                    std::vector<int>& owned_element_lids);
};

//---------------------------------------------------------------------------//
class RestartWriter : public Restart
{
  public:
    // NOTE: allow_dofmap_overwrite is false by default and should be provided
    //       only in unit tests.
    RestartWriter(const Teuchos::RCP<const panzer_stk::STK_Interface>& mesh,
                  const Teuchos::RCP<const panzer::GlobalIndexer>& dof_manager,
                  const Teuchos::ParameterList& output_params,
                  const bool allow_dofmap_overwrite = false);

    ~RestartWriter();

    void
    writeSolution(const Teuchos::RCP<const Thyra::VectorBase<double>>& x,
                  const Teuchos::RCP<const Thyra::VectorBase<double>>& x_dot,
                  const int index,
                  const double time = 0.0);

  private:
    Teuchos::RCP<const panzer::GlobalIndexer> _dof_manager;
    std::string _file_prefix;
    std::vector<int> _displacements;
    std::unordered_map<panzer::GlobalOrdinal, int> _global_to_local;
    MPI_Datatype _dof_type;
};

//---------------------------------------------------------------------------//
class RestartReader : public Restart
{
  public:
    RestartReader(const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
                  const Teuchos::ParameterList& input_params);

    void
    readSolution(const Teuchos::RCP<const panzer_stk::STK_Interface>& mesh,
                 const Teuchos::RCP<const panzer::GlobalIndexer>& dof_manager,
                 const Teuchos::RCP<Thyra::VectorBase<double>>& x,
                 const Teuchos::RCP<Thyra::VectorBase<double>>& x_dot);

    double initialStateTime() const { return _t_init; }

  private:
    std::string _restart_file_name;
    std::string _dofmap_file_name;
    double _t_init;

    void update_vector(const Teuchos::RCP<Thyra::VectorBase<double>>& vec,
                       const std::vector<double>& data) const;
};

//---------------------------------------------------------------------------//

} // end namespace Mesh
} // end namespace VertexCFD

#endif // end VERTEXCFD_MESH_RESTART_HPP

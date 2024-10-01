#ifndef VERTEXCFD_MESH_STKREADERFACTORY_HPP
#define VERTEXCFD_MESH_STKREADERFACTORY_HPP

#include <string>

#include <PanzerAdaptersSTK_config.hpp>
#include <Panzer_STK_MeshFactory.hpp>

#include <stk_io/StkMeshIoBroker.hpp>

namespace VertexCFD
{
namespace Mesh
{
//---------------------------------------------------------------------------//
/** External function to return the dimension of an Exodus or CGNS
 * mesh. This uses a quick temporary read of the meta data from input
 * file/mesh and is intended if you need the mesh dimension before the
 * MeshFactory is used to build the uncommitted mesh. Once
 * buildUncommittedMesh() is called, you can query the dimension from
 * the MetaData in the STK_Interface object (will be faster than
 * creating mesh metadata done in this function).
 *
 * \param[in] mesh_str Filename containing the mesh string, or the mesh string
 * itself.
 * \param[in] parallel_mach Descriptor for machine to build this mesh on.
 * \param[in] is_exodus Set to true for Exodus mesh, set to false for CGNS
 * mesh.
 *
 * \returns Integer indicating the spatial dimension of the mesh.
 */
int getMeshDimension(const std::string& mesh_str,
                     stk::ParallelMachine parallel_mach,
                     const bool is_exodus = true);

//---------------------------------------------------------------------------//
/** Concrete mesh factory instantiation. This reads
 * a mesh from an exodus file and builds a STK_Interface object.
 *
 * Also, if a nonzero restart index (the Exodus indices are 1 based) is
 * specified then this will set the initial state time in the STK_Interface.
 * However, as prescribed by that interface the currentStateTime is only
 * set by the writeToExodus call, thus the currentStateTime of the created
 * STK_Interface object will be zero. It is up to the user to rectify this
 * when calling writeToExodus.
 */
class StkReaderFactory : public panzer_stk::STK_MeshFactory
{
  public:
    StkReaderFactory();

    /** \brief Ctor
     *
     * \param[in] file_name Name of the input file.
     * \param[in] restart_index Index used for restarts.
     * \param[in] is_exodus If true, the input file is in exodus format. If
     * false, it assumes CGNS format.
     */
    StkReaderFactory(const std::string& file_name,
                     const int restart_index = 0,
                     const bool is_exodus = true);

    /** Construct a STK_Inteface object described by this factory.
     *
     * \param[in] parallel_mach Descriptor for machine to build this mesh on.
     *
     * \returns Pointer to <code>STK_Interface</code> object with
     *          <code>isModifiable()==false</code>.
     */
    virtual Teuchos::RCP<panzer_stk::STK_Interface>
    buildMesh(stk::ParallelMachine parallel_mach) const override;

    /** This builds all the meta data of the mesh. Does not call
     * metaData->commit. Allows user to add solution fields and other pieces.
     * The mesh can be "completed" by calling
     * <code>completeMeshConstruction</code>.
     */
    virtual Teuchos::RCP<panzer_stk::STK_Interface>
    buildUncommitedMesh(stk::ParallelMachine parallel_mach) const override;

    /** Finishes building a mesh object started by
     * <code>buildUncommitedMesh</code>.
     */
    virtual void
    completeMeshConstruction(panzer_stk::STK_Interface& mesh,
                             stk::ParallelMachine parallel_mach) const override;

    //! From ParameterListAcceptor. Must be called if the empty ctor is used to
    //! construct this object.
    void setParameterList(
        const Teuchos::RCP<Teuchos::ParameterList>& param_list) override;

    //! From ParameterListAcceptor
    Teuchos::RCP<const Teuchos::ParameterList>
    getValidParameters() const override;

    //! Get file name mean is read from.
    const std::string& getFileName() const { return file_name_; }

  protected:
    void registerElementBlocks(panzer_stk::STK_Interface& mesh,
                               stk::io::StkMeshIoBroker& mesh_data) const;
    void registerSidesets(panzer_stk::STK_Interface& mesh) const;
    void registerNodesets(panzer_stk::STK_Interface& mesh) const;

    std::string file_name_;
    std::string decomp_method_;
    int restart_index_;
    bool is_exodus_;

  private:
    //! Did the user request mesh scaling
    bool user_mesh_scaling_;

    //! If requested, scale the input mesh by this factor
    double mesh_scale_factor_;

    //! Number of levels of inline uniform mesh refinement to be applied to
    //! exodus mesh
    int levels_of_refinement_;
};

//---------------------------------------------------------------------------//

} // end namespace Mesh
} // end namespace VertexCFD

#endif // VERTEXCFD_STKREADERFACTORY_HPP

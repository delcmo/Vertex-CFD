#ifndef VERTEXCFD_TEMPUSOBSERVER_WRITEMATRIX_IMPL_HPP
#define VERTEXCFD_TEMPUSOBSERVER_WRITEMATRIX_IMPL_HPP

#include <string>
#include <vector>

#include <EpetraExt_MultiVectorOut.h>
#include <EpetraExt_RowMatrixOut.h>
#include <Epetra_CrsMatrix.h>
#include <MatrixMarket_Tpetra.hpp>
#include <NOX_Abstract_Group.H>
#include <NOX_Solver_Generic.H>
#include <NOX_Thyra_Vector.H>
#include <Panzer_NodeType.hpp>
#include <Thyra_Amesos2LinearOpWithSolve_decl.hpp>
#include <Thyra_AmesosLinearOpWithSolve.hpp>
#include <Thyra_AztecOOLinearOpWithSolve.hpp>
#include <Thyra_BelosLinearOpWithSolve.hpp>
#include <Thyra_DefaultSpmdVector.hpp>
#include <Thyra_EpetraLinearOp.hpp>
#include <Thyra_EpetraThyraWrappers.hpp>
#include <Thyra_NonlinearSolver_NOX.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
WriteMatrix<Scalar>::WriteMatrix(const Teuchos::ParameterList& write_params)
{
    // Determine which time steps will write out matrix files
    if (write_params.isType<Teuchos::Array<int>>("Write Steps"))
    {
        _write_steps
            = write_params.template get<Teuchos::Array<int>>("Write Steps");
    }
    else
    {
        throw std::runtime_error(
            "Requested matrix output but 'Write Steps' was not provided'");
    }

    // Determine if we want the residual in addition to matrix
    _write_residual = false;
    if (write_params.isType<bool>("Write Residual"))
    {
        _write_residual = write_params.template get<bool>("Write Residual");
    }

    // Set base filename for files. If not specified, use generic
    // "jacobian" and "residual"
    if (write_params.isType<std::string>("Matrix File Prefix"))
    {
        auto base_prefix
            = write_params.template get<std::string>("Matrix File Prefix");
        _jacobian_prefix = base_prefix + "-jacobian";
        _residual_prefix = base_prefix + "-residual";
    }
    else
    {
        _jacobian_prefix = "jacobian";
        _residual_prefix = "residual";
    }
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::observeStartIntegrator(
    const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::observeStartTimeStep(const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::observeNextTimeStep(const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::observeBeforeTakeStep(const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::observeAfterTakeStep(const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::observeAfterCheckTimeStep(
    const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::observeEndTimeStep(
    const Tempus::Integrator<Scalar>& integrator)
{
    auto step_index = integrator.getIndex();

    // Only write solution at specified time steps
    if (std::find(_write_steps.begin(), _write_steps.end(), step_index)
        != _write_steps.end())
    {
        // Determine filename for this step
        std::string filename = _jacobian_prefix + "-"
                               + std::to_string(step_index) + ".mtx";

        // Message to put into comments of file
        std::string descr = "VertexCFD Jacobian matrix at time step "
                            + std::to_string(step_index);

        // Get nonlinear solver from time integrator
        auto stepper = integrator.getStepper();
        auto nonlinear_solver = stepper->getSolver();
        if (!nonlinear_solver)
        {
            throw std::runtime_error("Nonlinear solver not available");
        }

        // Extract linear solver operator
        auto linear_solver = nonlinear_solver->get_W();
        if (!linear_solver)
        {
            throw std::runtime_error("Linear solver is not available");
        }

        // Extra linear operator (Jacobian), toggling on Belos vs. AztecOO
        auto jacobian_op = this->extractLinearOp(linear_solver);

        // Process matrix output, toggling on Epetra vs. Tpetra
        this->write_jacobian(jacobian_op, filename, descr);

        if (_write_residual)
        {
            filename = _residual_prefix + "-" + std::to_string(step_index)
                       + ".mtx";
            descr = "Residual vector at time step "
                    + std::to_string(step_index);

            // Extract NOX solver
            auto thyranox_solver
                = Teuchos::rcp_dynamic_cast<const Thyra::NOXNonlinearSolver>(
                    nonlinear_solver);
            auto nox_solver = thyranox_solver->getNOXSolver();

            // Get residual vector from the solver
            // Use the _previous_ solution group so that the residual
            // is the RHS from the last nonlinear iteration rather than
            // the converged residual after the solve is complete.
            const auto& group = nox_solver->getPreviousSolutionGroup();
            const auto& nox_resid = group.getF();
            const NOX::Thyra::Vector& noxthyra_resid
                = dynamic_cast<const NOX::Thyra::Vector&>(nox_resid);
            auto thyra_resid = noxthyra_resid.getThyraRCPVector();

            // Process vector output, toggling on Epetra vs. Tpetra
            this->write_residual(thyra_resid, filename, descr);
        }
    }
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::observeEndIntegrator(const Tempus::Integrator<Scalar>&)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
Teuchos::RCP<const Thyra::LinearOpBase<Scalar>>
WriteMatrix<Scalar>::extractLinearOp(
    const Teuchos::RCP<const Thyra::LinearOpWithSolveBase<Scalar>>&
        linear_solver) const
{
    Teuchos::RCP<const Thyra::LinearOpBase<Scalar>> jacobian_op;

    // Currently support Belos, AztecOO, Amesos, Amesos2 solvers.
    // These all implement the same "getOp" interface, but that method is
    //  NOT a member of the LinearOpWithSolve base class.
    // Try casting to each of these in turn to extract the Jacobian
    jacobian_op
        = this->tryExtractForwardOp<Thyra::BelosLinearOpWithSolve<double>>(
            linear_solver);
    if (jacobian_op)
        return jacobian_op;

    jacobian_op = this->tryExtractForwardOp<Thyra::AztecOOLinearOpWithSolve>(
        linear_solver);
    if (jacobian_op)
        return jacobian_op;

    jacobian_op = this->tryExtractForwardOp<Thyra::AmesosLinearOpWithSolve>(
        linear_solver);
    if (jacobian_op)
        return jacobian_op;

    jacobian_op
        = this->tryExtractForwardOp<Thyra::Amesos2LinearOpWithSolve<double>>(
            linear_solver);
    if (jacobian_op)
        return jacobian_op;

    throw std::runtime_error(
        "Unrecognized linear solver type in matrix output");
}

//---------------------------------------------------------------------------//
// Attempt to extract the Jacobian from a solver of the type specified by
// the SolverType template parameter. If the concrete solver is not of the
// specified type, a null RCP will be returned.
//---------------------------------------------------------------------------//
template<class Scalar>
template<class SolverType>
Teuchos::RCP<const Thyra::LinearOpBase<Scalar>>
WriteMatrix<Scalar>::tryExtractForwardOp(
    const Teuchos::RCP<const Thyra::LinearOpWithSolveBase<Scalar>>&
        linear_solver) const
{
    Teuchos::RCP<const Thyra::LinearOpBase<Scalar>> jacobian_op;

    auto derived_solver
        = Teuchos::rcp_dynamic_cast<const SolverType>(linear_solver);
    if (derived_solver)
    {
        // Const-cast and extract forward source operator (which is the
        // Jacobian)
        auto nonconst_derived_solver
            = Teuchos::rcp_const_cast<SolverType>(derived_solver);
        auto fwd_opsrc = nonconst_derived_solver->extract_fwdOpSrc();
        jacobian_op = fwd_opsrc->getOp();
    }

    return jacobian_op;
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::write_jacobian(
    const Teuchos::RCP<const Thyra::LinearOpBase<Scalar>>& jacobian_op,
    const std::string& filename,
    const std::string& description) const
{
    // Toggle on Epetra vs. Tpetra
    auto jacobian_epetra
        = Teuchos::rcp_dynamic_cast<const Thyra::EpetraLinearOp>(jacobian_op);
    if (jacobian_epetra)
    {
        auto epetra_op = jacobian_epetra->epetra_op();
        auto epetra_mat
            = Teuchos::rcp_dynamic_cast<const Epetra_RowMatrix>(epetra_op);
        if (!epetra_mat)
        {
            throw std::runtime_error("Epetra operator is not a RowMatrix");
        }

        EpetraExt::RowMatrixToMatrixMarketFile(
            filename.c_str(), *epetra_mat, description.c_str());
    }
    else
    {
        // If not Epetra, must be Tpetra
        auto jacobian_tpetra = Teuchos::rcp_dynamic_cast<
            const Thyra::TpetraLinearOp<Scalar,
                                        int,
                                        panzer::GlobalOrdinal,
                                        panzer::TpetraNodeType>>(jacobian_op);
        if (!jacobian_tpetra)
        {
            throw std::runtime_error("Unexpected matrix type");
        }

        // Tpetra matrix output requires concrete type (CrsMatrix, not
        // RowMatrix)
        using TpetraMatrix = Tpetra::
            CrsMatrix<Scalar, int, panzer::GlobalOrdinal, panzer::TpetraNodeType>;
        auto tpetra_op = jacobian_tpetra->getConstTpetraOperator();
        auto tpetra_mat
            = Teuchos::rcp_dynamic_cast<const TpetraMatrix>(tpetra_op);
        if (!tpetra_mat)
        {
            throw std::runtime_error("Tpetra operator is not a CrsMatrix");
        }

        Tpetra::MatrixMarket::Writer<TpetraMatrix>::writeSparseFile(
            filename, *tpetra_mat, "", description);
    }
}

//---------------------------------------------------------------------------//
template<class Scalar>
void WriteMatrix<Scalar>::write_residual(
    const Teuchos::RCP<const Thyra::VectorBase<Scalar>>& vec,
    const std::string& filename,
    const std::string& description) const
{
    // Vector should be a DefaultSpmdVector (Epetra) or a TpetraVector.
    auto spmd_vec
        = Teuchos::rcp_dynamic_cast<const Thyra::DefaultSpmdVector<double>>(
            vec);
    if (spmd_vec)
    {
        auto space = vec->space();
        auto epetra_map = Thyra::get_Epetra_Map(space);
        auto epetra_vec = Thyra::get_Epetra_Vector(vec, epetra_map);

        EpetraExt::MultiVectorToMatrixMarketFile(
            filename.c_str(), *epetra_vec, "", description.c_str());
    }

    auto thyratpetra_vec = Teuchos::rcp_dynamic_cast<
        const Thyra::
            TpetraVector<double, int, panzer::GlobalOrdinal, panzer::TpetraNodeType>>(
        vec);
    if (thyratpetra_vec)
    {
        auto tpetra_vec = thyratpetra_vec->getConstTpetraVector();

        using TpetraMatrix = Tpetra::
            CrsMatrix<Scalar, int, panzer::GlobalOrdinal, panzer::TpetraNodeType>;
        Tpetra::MatrixMarket::Writer<TpetraMatrix>::writeDenseFile(
            filename, *tpetra_vec, "", description);
    }
}

//---------------------------------------------------------------------------//

} // end namespace TempusObserver
} // end namespace VertexCFD

#endif // end VERTEXCFD_TEMPUSOBSERVER_WRITEMATRIX_IMPL_HPP

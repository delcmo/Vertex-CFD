#ifndef VERTEXCFD_LINEARSOLVERS_CUSOLVERNONPUBLIC_HPP
#define VERTEXCFD_LINEARSOLVERS_CUSOLVERNONPUBLIC_HPP

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#include <cassert>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/*
 * Prototypes not in public header file
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrluConfigHost(csrluInfoHost_t info,
                                                        int reorder /* 0 or 1
                                                                     */
);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlucondHost(csrluInfoHost_t info,
                                                      double* max_diag_u_host,
                                                      double* min_diag_u_host,
                                                      double* max_l_host);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrluNnzMHost(cusolverSpHandle_t handle,
                                                      int* nnz_m_ref_host,
                                                      csrluInfoHost_t info);

cusolverStatus_t CUSOLVERAPI
cusolverSpDcsrluExtractMHost(cusolverSpHandle_t handle,
                             int* P_host,
                             int* Q_host,
                             const cusparseMatDescr_t M_descr,
                             double* M_values_host,
                             int* M_rowptr_host,
                             int* M_colind_host,
                             csrluInfoHost_t info,
                             void* work_host);

struct csrgluInfo;
typedef struct csrgluInfo* csrgluInfo_t;

cusolverStatus_t CUSOLVERAPI cusolverSpCreateGluInfo(csrgluInfo_t* info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyGluInfo(csrgluInfo_t info);

cusolverStatus_t CUSOLVERAPI
cusolverSpDgluSetup(cusolverSpHandle_t handle,
                    int m,
                    int nnzA,
                    const cusparseMatDescr_t A_descr,
                    const int* A_rowptr_host,
                    const int* A_colind_host,
                    const int* P_host,
                    const int* Q_host,
                    int M_nnz,
                    const cusparseMatDescr_t M_descr,
                    const int* M_rowptr_host,
                    const int* M_colind_host,
                    csrgluInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluBufferSize(
    cusolverSpHandle_t handle, csrgluInfo_t info, size_t* p_buffer_size_host);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluAnalysis(cusolverSpHandle_t handle,
                                                    csrgluInfo_t info,
                                                    void* work);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluReset(cusolverSpHandle_t handle,
                                                 int num_rows,
                                                 int A_nnz,
                                                 const cusparseMatDescr_t A_descr,
                                                 const double* A_values,
                                                 const int* A_rowptr,
                                                 const int* A_colind,
                                                 csrgluInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluFactor(cusolverSpHandle_t handle,
                                                  csrgluInfo_t info,
                                                  void* work);

cusolverStatus_t CUSOLVERAPI cusolverSpDgluSolve(cusolverSpHandle_t handle,
                                                 int num_rows,
                                                 int A_nnz,
                                                 const cusparseMatDescr_t A_descr,
                                                 const double* A_values,
                                                 const int* A_rowptr,
                                                 const int* A_colind,
                                                 const double* b, /* right hand
                                                                     side */
                                                 double* x, /* left hand side
                                                             */
                                                 int* ite_refine_succ_host,
                                                 double* r_nrminf_ptr_host,
                                                 csrgluInfo_t info,
                                                 void* work);

cusolverStatus_t CUSOLVERAPI cusolverSpDnrminf(cusolverSpHandle_t handle,
                                               int n,
                                               const double* x,
                                               double* result_host, /* |x|_inf,
                                                                     * host
                                                                     */
                                               void* work /* at least 8192
                                                               bytes */
);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // VERTEXCFD_LINEARSOLVERS_CUSOLVERNONPUBLIC_CUH

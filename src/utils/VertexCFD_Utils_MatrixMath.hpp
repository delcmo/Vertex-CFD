#ifndef VERTEXCFD_UTILS_MATRIXMATH_HPP
#define VERTEXCFD_UTILS_MATRIXMATH_HPP

#include "VertexCFD_Utils_Constants.hpp"
#include "VertexCFD_Utils_TypeTraits.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>

namespace VertexCFD
{
namespace MatrixMath
{
//---------------------------------------------------------------------------//
// LU decomposition (partial pivoting).
// Finds permutation p and rewrites matrix A so that A[[p]] = LU
// Input matrix A must have full rank or division by zero pivot may occur
//---------------------------------------------------------------------------//
template<typename Matrix, typename Permutation>
KOKKOS_INLINE_FUNCTION void LUP(Matrix& A, Permutation& p)
{
    using std::abs;

    const int N = p.extent(0);

    using value_type = typename Matrix::value_type;

    // Initialize permutation vector
    for (int i = 0; i < N; ++i)
        p(i) = i;

    for (int k = 0; k < N; ++k)
    {
        int pivot = k;
        // max is only used for finding the pivot so doesn't need derivatives.
        double max = abs(Sacado::ScalarValue<value_type>::eval(A(p(pivot), k)));

        for (int i = k + 1; i < N; ++i)
        {
            const int row = p(i);
            // temp is only used for finding the pivot so doesn't need
            // derivatives.
            const double temp
                = abs(Sacado::ScalarValue<value_type>::eval(A(row, k)));
            if (temp > max)
            {
                max = temp;
                pivot = i;
            }
        }

        if (p(pivot) != p(k))
        {
            const int swap = p(pivot);
            p(pivot) = p(k);
            p(k) = swap;
        }

        pivot = p(k);
        for (int i = k + 1; i < N; ++i)
        {
            const int row = p(i);
            A(row, k) /= A(pivot, k);
            for (int j = k + 1; j < N; ++j)
                A(row, j) -= A(row, k) * A(pivot, j);
        }
    }
}

//---------------------------------------------------------------------------//
// LU solve (partial pivoting).
// Takes matrix A and permutation p such that A[[p]] = LU and solves Ax = b
// Solution vector is stored in b upon completion.
//---------------------------------------------------------------------------//
template<typename Matrix, typename Permutation, typename Vector>
KOKKOS_INLINE_FUNCTION void
LUP_solve(const Matrix& LU, const Permutation& p, Vector& work, Vector& b)
{
    const int N = p.extent(0);

    for (int i = 0; i < N; ++i)
        work(i) = b(i);

    for (int i = 1; i < N; ++i)
        for (int j = 0; j < i; ++j)
            work(p(i)) -= LU(p(i), j) * work(p(j));

    for (int i = N - 1; i >= 0; --i)
    {
        for (int j = i + 1; j < N; ++j)
            work(p(i)) -= LU(p(i), j) * work(p(j));

        work(p(i)) /= LU(p(i), i);
    }

    for (int i = 0; i < N; ++i)
        b(i) = work(p(i));
}
//---------------------------------------------------------------------------//

} // end namespace MatrixMath
} // end namespace VertexCFD

#endif // end VERTEXCFD_UTILS_MATRIXMATH_HPP

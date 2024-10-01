#ifndef VERTEXCFD_UTILS_SMOOTHMATH_HPP
#define VERTEXCFD_UTILS_SMOOTHMATH_HPP

#include "VertexCFD_Utils_Constants.hpp"
#include "VertexCFD_Utils_TypeTraits.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>

namespace VertexCFD
{
namespace SmoothMath
{
//---------------------------------------------------------------------------//
template<typename T>
KOKKOS_INLINE_FUNCTION ResultType<T> abs(const T& x, const double tol)
{
    if (x >= tol)
    {
        return x;
    }
    else if (x <= -tol)
    {
        return -x;
    }
    else
    {
        return 0.5 * (x * x / tol + tol);
    }
}

//---------------------------------------------------------------------------//
template<typename T1, typename T2>
KOKKOS_INLINE_FUNCTION ResultType<T1, T2>
max(const T1& x, const T2& y, const double tol)
{
    if (tol == 0.0)
    {
        if (x < y)
        {
            return y;
        }
        else
        {
            return x;
        }
    }
    else
    {
        return 0.5 * (x + y + abs(x - y, tol));
    }
}

//---------------------------------------------------------------------------//
template<typename T1, typename T2>
KOKKOS_INLINE_FUNCTION ResultType<T1, T2>
min(const T1& x, const T2& y, const double tol)
{
    if (tol == 0.0)
    {
        if (x > y)
        {
            return y;
        }
        else
        {
            return x;
        }
    }
    else
    {
        return 0.5 * (x + y - abs(x - y, tol));
    }
}

//---------------------------------------------------------------------------//
template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION ResultType<T1, T2, T3>
clamp(const T1& x, const T2& lo, const T3& hi, const double tol)
{
    // Note: This could be implemented as min(max(x, lo, tol), hi, tol) or
    // similar, but that would result in more temporary/intermediate values
    // being allocated.

    // -inf < x < inf
    if (x <= lo - tol)
    {
        return lo;
    }

    // lo - tol < x < inf
    else if (x >= hi + tol)
    {
        return hi;
    }

    // lo - tol < x < hi + tol
    else if (x < lo + tol)
    {
        // lo - tol < x < lo + tol => -tol < (x-lo) < tol
        return 0.5 * (x + lo + 0.5 * ((x - lo) * (x - lo) / tol + tol));
    }

    // lo + tol <= x < hi + tol
    else if (x > hi - tol)
    {
        // hi - tol < x < hi + tol => -tol < (x-hi) < tol
        return 0.5 * (x + hi - 0.5 * ((x - hi) * (x - hi) / tol + tol));
    }

    // lo + tol <= x <= h - tol
    else
    {
        return x;
    }
}

//---------------------------------------------------------------------------//
template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION ResultType<T1, T2, T3>
ramp(const T1& x, const T2& start, const T3& end)
{
    using std::sin;
    constexpr double half_pi = 0.5 * Constants::pi;
    if (x <= start)
    {
        return 0.0;
    }
    else if (x >= end)
    {
        return 1.0;
    }
    else
    {
        return 0.5
               * (sin(half_pi * (2.0 * x - (start + end)) / (end - start))
                  + 1.0);
    }
}

//---------------------------------------------------------------------------//
template<typename T1, typename T2>
KOKKOS_INLINE_FUNCTION ResultType<T1, T2>
hypot(const T1& x, const T2& y, const double tol)
{
    using std::sqrt;

    const ResultType<T1, T2> dotp = x * x + y * y;
    const double tol2 = tol * tol;

    if (tol2 <= dotp)
    {
        return sqrt(dotp);
    }
    else
    {
        return 0.5 * (dotp + tol2) / tol;
    }
}

//---------------------------------------------------------------------------//
template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION ResultType<T1, T2, T3>
hypot(const T1& x, const T2& y, const T3& z, const double tol)
{
    using std::sqrt;

    const ResultType<T1, T2, T3> dotp = x * x + y * y + z * z;
    const double tol2 = tol * tol;

    if (tol2 <= dotp)
    {
        return sqrt(dotp);
    }
    else
    {
        return 0.5 * (dotp + tol2) / tol;
    }
}

template<typename T1, typename ReturnType = ResultType<typename T1::value_type>>
KOKKOS_INLINE_FUNCTION ReturnType norm(const T1& v, const double tol)
{
    using std::sqrt;

    using scalar_type = std::remove_cv_t<ReturnType>;
    scalar_type dotp = 0.0;
    int num_space_dim = v.size();

    for (int i = 0; i < num_space_dim; ++i)
    {
        dotp += v[i] * v[i];
    }

    const double tol2 = tol * tol;
    if (tol2 <= dotp)
    {
        return sqrt(dotp);
    }
    else
    {
        return 0.5 * (dotp + tol2) / tol;
    }
}

//---------------------------------------------------------------------------//
template<typename T1,
         typename T2,
         typename ReturnType
         = ResultType<typename T1::value_type, typename T2::value_type>>
KOKKOS_INLINE_FUNCTION ReturnType norm(const T1& v,
                                       const T2& M,
                                       const double tol)
{
    using std::sqrt;

    using scalar_type = std::remove_cv_t<ReturnType>;
    scalar_type xi = 0.0;
    scalar_type row_sum;

    int num_space_dim = M.extent(0);

    for (int i = 0; i < num_space_dim; ++i)
    {
        row_sum = 0.0;
        for (int j = 0; j < num_space_dim; ++j)
        {
            row_sum += M(i, j) * v[j];
        }
        xi += v[i] * row_sum;
    }

    const double tol2 = tol * tol;
    if (tol2 <= xi)
    {
        return sqrt(xi);
    }
    else
    {
        return 0.5 * (xi + tol2) / tol;
    }
}

//---------------------------------------------------------------------------//

} // end namespace SmoothMath
} // end namespace VertexCFD

#endif // end VERTEXCFD_UTILS_SMOOTHMATH_HPP

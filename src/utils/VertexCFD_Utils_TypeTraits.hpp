#ifndef VERTEXCFD_TYPETRAITS_HPP
#define VERTEXCFD_TYPETRAITS_HPP

#include <Sacado_Traits.hpp>

namespace VertexCFD
{
namespace Utils
{
namespace Impl
{
//---------------------------------------------------------------------------//
// Get the resulting scalar type from a combination of one or more types which
// may be standard arithmetic types, AD scalars, or AD expression templates.
//
// This may be used as the return type of a generic function of one or more
// arguments and will ensure that a valid type is returned.
//---------------------------------------------------------------------------//
template<typename... Types>
struct ResultType;

//---------------------------------------------------------------------------//
// For a single type, forward to Sacado type trait to get the base type of an
// expression.
template<typename Type>
struct ResultType<Type>
{
    using type = typename Sacado::BaseExprType<Type>::type;
};

//---------------------------------------------------------------------------//
// For two types, forward to Sacado type trait for the promoted type of a
// binary operation.
template<typename Type1, typename Type2>
struct ResultType<Type1, Type2>
{
    using type = typename Sacado::Promote<Type1, Type2>::type;
};

//---------------------------------------------------------------------------//
// For more than two types, apply the above recursively to get a single result
// type.
template<typename Type1, typename Type2, typename... Types>
struct ResultType<Type1, Type2, Types...>
{
    using type =
        typename ResultType<typename Sacado::Promote<Type1, Type2>::type,
                            Types...>::type;
};
//---------------------------------------------------------------------------//
} // namespace Impl
} // namespace Utils

//---------------------------------------------------------------------------//
// User-facing type alias for all of the above.
// Since the primary use case is for return types of generic functions, add
// this alias to the VertexCFD namespace to avoid unnecessary verbosity.
template<typename... Types>
using ResultType = typename Utils::Impl::ResultType<Types...>::type;

} // namespace VertexCFD

#endif

#ifndef NOARR_STRUCTURES_STRUCT_TRAITS_HPP
#define NOARR_STRUCTURES_STRUCT_TRAITS_HPP

#include <type_traits>

#include "std_ext.hpp"
#include "struct_decls.hpp"
#include "scalar.hpp"

namespace noarr {

template<typename T, typename = void>
struct _is_static_dimension {
    using type = std::false_type;
};

template<typename T>
struct _is_static_dimension<T, void_t<decltype(std::declval<T>().template offset<std::size_t(0)>())>> {
    using type = std::true_type;
};

template<typename T, typename = void>
struct _is_dynamic_dimension {
    using type = std::false_type;
};

template<typename T, std::size_t O, typename = void>
struct _has_static_offset {
    using value_type = bool;
    constexpr static value_type value = false;
};

template<typename T, std::size_t O>
struct _has_static_offset<T, O, void_t<decltype(static_cast<std::size_t (T::*)(std::size_t) const>(&T::template offset<O>))>> {
    using value_type = bool;
    constexpr static value_type value = true;
};

template<typename T, typename = void>
struct _has_dynamic_offset {
    using value_type = bool;
    constexpr static value_type value = false;
};

template<typename T>
struct _has_dynamic_offset<T, void_t<decltype(static_cast<std::size_t (T::*)(std::size_t) const>(&T::offset))>> {
    using value_type = bool;
    constexpr static value_type value = true;
};

template<typename T>
struct _is_dynamic_dimension<T, void_t<decltype(std::declval<T>().offset(std::declval<std::size_t>()))>> {
    using type = std::true_type;
};

template<typename T>
using is_static_dimension = std::enable_if_t<_has_static_offset<T, 0>::value, typename _is_static_dimension<T>::type>;

template<typename T>
using is_dynamic_dimension = std::enable_if_t<_has_dynamic_offset<T>::value, typename _is_dynamic_dimension<T>::type>;

template<typename T, typename = void>
struct _is_point;

template<typename T>
using is_point = typename _is_point<remove_cvref<T>>::type;

template<typename T>
struct _is_point<T, std::enable_if_t<(std::is_same<typename get_struct_desc_t<T>::dims, dims_impl<>>::value) && tuple_forall<is_point, typename sub_structures<T>::value_type>::value>> {
    using type = std::true_type;
};

template<typename T, typename = void>
struct _is_cube {
    using type = std::false_type;
};

template<typename T>
using is_cube = typename _is_cube<remove_cvref<T>>::type;

template<typename T>
struct _is_cube<T, std::enable_if_t<!std::is_same<typename get_struct_desc_t<T>::dims, dims_impl<>>::value && is_dynamic_dimension<T>::value && tuple_forall<is_cube, typename sub_structures<T>::value_type>::value>> {
    using type = std::true_type;
};

template<typename T>
struct _is_cube<T, std::enable_if_t<std::is_same<typename get_struct_desc_t<T>::dims, dims_impl<>>::value && tuple_forall<is_cube, typename sub_structures<T>::value_type>::value>> {
    using type = std::true_type;
};

template<typename T>
struct _is_scalar {
    using type = std::false_type;
};

template<typename T>
using is_scalar = typename _is_scalar<T>::type;

template<typename T>
struct _is_scalar<scalar<T>> {
    using type = std::true_type;
};

template<typename T, typename = void>
struct _scalar_t;

template<typename T>
using scalar_t = typename _scalar_t<T>::type;

template<typename T>
struct _scalar_t<T, std::enable_if_t<!is_scalar<T>::value && is_cube<T>::value>> {
    using type = scalar_t<typename T::template get_t<>>;
};

template<typename T>
struct _scalar_t<scalar<T>> {
    using type = T;
};

}

#endif // NOARR_STRUCTURES_STRUCT_TRAITS_HPP

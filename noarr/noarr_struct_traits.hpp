#ifndef NOARR_STRUCT_TRAITS_HPP
#define NOARR_STRUCT_TRAITS_HPP

#include <type_traits>

#include "noarr_std_ext.hpp"
#include "noarr_core.hpp"
#include "noarr_struct_desc.hpp"
#include "noarr_scalar.hpp"

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

template<typename T>
struct _is_dynamic_dimension<T, void_t<decltype(std::declval<T>().offset(std::declval<std::size_t>()))>> {
    using type = std::true_type;
};

template<typename T>
using is_static_dimension = typename _is_static_dimension<T>::type;

template<typename T>
using is_dynamic_dimension = typename _is_dynamic_dimension<T>::type;

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
struct _is_cube<T, std::enable_if_t<(is_dynamic_dimension<T>::value || std::is_same<typename get_struct_desc_t<T>::dims, dims_impl<>>::value) && tuple_forall<is_cube, typename sub_structures<T>::value_type>::value>> {
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

#endif // NOARR_STRUCT_TRAITS_HPP

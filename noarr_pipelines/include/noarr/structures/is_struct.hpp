#ifndef NOARR_IS_STRUCT_HPP
#define NOARR_IS_STRUCT_HPP

#include "noarr_std_ext.hpp"
#include "noarr_core.hpp"

// TODO: is_struct and is_structoid should be more strict

namespace noarr {

template<typename T, typename = void>
struct _is_struct {
    using type = std::false_type;
};

template<typename T, typename = void>
struct _is_structoid {
    using type = std::false_type;
};

template<typename T, typename = void>
struct _has_construct {
    using type = std::false_type;
};

template<typename T>
using has_construct = typename _has_construct<T>::type;

template<typename T, typename = void>
struct _has_get_t {
    using type = std::false_type;
};

template<typename T>
using has_get_t = typename _has_get_t<T>::type;

template<typename T, typename = void>
struct _has_get_t1 {
    using type = std::false_type;
};

template<typename T, typename = void>
struct _has_get_t2 {
    using type = std::false_type;
};

template<typename T, typename = void>
struct _has_offset;

template<typename T, typename = void>
struct _has_size;

template<typename T>
using is_struct = typename _is_struct<T>::type;

template<typename T>
using is_structoid = typename _is_structoid<T>::type;

template<typename T>
struct _has_construct<T, void_t<decltype(construct(std::declval<T>(), std::declval<typename sub_structures<T>::value_type>()))>> {
    using type = std::true_type;
};

template<typename T>
struct _has_get_t1<T, void_t<typename T::template get_t<>>> {
    using type = std::true_type;
};

template<typename T>
struct _has_get_t2<T, void_t<typename T::template get_t<std::integral_constant<std::size_t, 0>>>> {
    using type = std::true_type;
};

template<typename T>
struct _has_get_t<T, std::enable_if_t<_has_get_t1<T>::type::value || _has_get_t2<T>::type::value>> {
    using type = std::true_type;
};

template<typename T>
struct _is_structoid<T, std::enable_if_t<has_construct<T>::value>> {
    using type = std::true_type;
};

template<typename T>
struct _is_struct<T, std::enable_if_t<is_structoid<T>::value && has_get_t<T>::value>> {
    using type = std::true_type;
};

}

#endif // NOARR_IS_STRUCT_HPP
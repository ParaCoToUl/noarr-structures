#ifndef NOARR_STRUCTURES_STRUCT_TRAITS_HPP
#define NOARR_STRUCTURES_STRUCT_TRAITS_HPP

#include <type_traits>

#include "std_ext.hpp"
#include "struct_decls.hpp"
#include "scalar.hpp"

namespace noarr {

namespace helpers {

template<typename T, typename = void>
struct is_static_dimension_impl {
    using type = std::false_type;
};

template<typename T>
struct is_static_dimension_impl<T, void_t<decltype(std::declval<T>().template offset<std::size_t(0)>())>> {
    using type = std::true_type;
};

template<typename T, typename = void>
struct is_dynamic_dimension_impl {
    using type = std::false_type;
};

template<typename T, std::size_t O, typename = void>
struct has_static_offset_impl {
    using value_type = bool;
    constexpr static value_type value = false;
};

template<typename T, std::size_t O>
struct has_static_offset_impl<T, O, void_t<decltype(static_cast<std::size_t (T::*)(std::size_t) const>(&T::template offset<O>))>> {
    using value_type = bool;
    constexpr static value_type value = true;
};

template<typename T, typename = void>
struct has_dynamic_offset_impl {
    using value_type = bool;
    constexpr static value_type value = false;
};

template<typename T>
struct has_dynamic_offset_impl<T, void_t<decltype(static_cast<std::size_t (T::*)(std::size_t) const>(&T::offset))>> {
    using value_type = bool;
    constexpr static value_type value = true;
};

template<typename T>
struct is_dynamic_dimension_impl<T, void_t<decltype(std::declval<T>().offset(std::declval<std::size_t>()))>> {
    using type = std::true_type;
};

template<typename T, typename = void>
struct is_point_impl;

} // namespace helpers

/**
 * @brief returns whether the structure has a static dimension (accepts static indices)
 * 
 * @tparam T: the structure
 */
template<typename T>
using is_static_dimension = std::conditional_t<helpers::has_static_offset_impl<T, 0>::value, typename helpers::is_static_dimension_impl<T>::type, std::false_type>;

/**
 * @brief returns whether the structure has a dynamic dimension (accepts dynamic indices)
 * 
 * @tparam T: the structure
 */
template<typename T>
using is_dynamic_dimension = std::conditional_t<helpers::has_dynamic_offset_impl<T>::value, typename helpers::is_dynamic_dimension_impl<T>::type, std::false_type>;

/**
 * @brief returns whether the structure is a point (a structure with no dimensions, or with all dimensions being fixed)
 * 
 * @tparam T: the structure
 */
template<typename T>
using is_point = typename helpers::is_point_impl<remove_cvref<T>>::type;

namespace helpers {

template<typename T>
struct is_point_impl<T, std::enable_if_t<(std::is_same<typename get_struct_desc_t<T>::dims, dims_impl<>>::value) && tuple_forall<is_point, typename sub_structures<T>::value_type>::value>> {
    using type = std::true_type;
};

template<typename T, typename = void>
struct is_cube_int_impl {
    using type = std::false_type;
};

template<typename T>
struct is_cube_impl {
    using type = typename is_cube_int_impl<T>::type;
};

} // namespace helpers

/**
 * @brief returns whether a structure is a cube (its dimension and dimension of its substructures, recursively, are all dynamic)
 * 
 * @tparam T: the structure
 */
template<typename T>
using is_cube = typename helpers::is_cube_impl<remove_cvref<T>>::type;

namespace helpers {

template<typename T>
struct is_cube_int_impl<T, std::enable_if_t<!std::is_same<typename get_struct_desc_t<T>::dims, dims_impl<>>::value && tuple_forall<is_cube, typename sub_structures<T>::value_type>::value>> {
    using type = std::conditional_t<is_dynamic_dimension<T>::value, std::true_type, std::false_type>;
};

template<typename T>
struct is_cube_int_impl<T, std::enable_if_t<std::is_same<typename get_struct_desc_t<T>::dims, dims_impl<>>::value && tuple_forall<is_cube, typename sub_structures<T>::value_type>::value>> {
    using type = std::true_type;
};

template<typename T>
struct is_scalar_impl {
    using type = std::false_type;
};

template<typename T>
struct is_scalar_impl<scalar<T>> {
    using type = std::true_type;
};

template<typename T, typename = void>
struct scalar_t_impl;

} // namespace helpers

/**
 * @brief returns whether a structure is a `scalar<...>`
 * 
 * @tparam T: the structure
 */
template<typename T>
using is_scalar = typename helpers::is_scalar_impl<T>::type;

/**
 * @brief returns the type of the value described by a `scalar<...>`
 * 
 * @tparam T: the `scalar<...>`
 */
template<typename T>
using scalar_t = typename helpers::scalar_t_impl<T>::type;

namespace helpers {

template<typename T>
struct scalar_t_impl<T, std::enable_if_t<!is_scalar<T>::value && is_cube<T>::value>> {
    using type = scalar_t<typename T::template get_t<>>;
};

template<typename T>
struct scalar_t_impl<scalar<T>> {
    using type = T;
};

}

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_TRAITS_HPP

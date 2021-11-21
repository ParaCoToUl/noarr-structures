#ifndef NOARR_STRUCTURES_IS_STRUCT_HPP
#define NOARR_STRUCTURES_IS_STRUCT_HPP

#include "std_ext.hpp"
#include "pipes.hpp"

namespace noarr {

namespace helpers {

template<class T, class = void>
struct is_struct_impl : std::false_type {};

template<class T, class = void>
struct is_structoid_impl : std::false_type {};

template<class T, class = void>
struct has_construct_impl : std::false_type {};

template<class T, class = void>
struct has_get_t_impl : std::false_type {};

template<class T, class = void>
struct has_get_t1_impl : std::false_type {};

template<class T, class = void>
struct has_get_t2_impl : std::false_type {};

template<class T, class = void>
struct has_size_impl;

} // namespace helpers

/**
 * @brief returns whether the type `T` has a suitable `construct` method
 * 
 * @tparam T: the input type
 */
template<class T>
using has_construct = helpers::has_construct_impl<T>;

/**
 * @brief returns whether the type `T` has a `get_t` member typedef
 * 
 * @tparam T: the input type
 */
template<class T>
using has_get_t = helpers::has_get_t_impl<T>;

/**
 * @brief returns whether the type `T` meets the criteria for structures
 * 
 * @tparam T: the input type
 */
template<class T>
using is_struct = helpers::is_struct_impl<T>;

/**
 * @brief returns whether the type `T` meets the criteria for structoids
 * 
 * @tparam T: the input type
 */
template<class T>
using is_structoid = helpers::is_structoid_impl<T>;

namespace helpers {

template<class T>
struct has_construct_impl<T, void_t<decltype(construct(std::declval<T>(), std::declval<typename sub_structures<T>::value_type>()))>> : std::true_type {};

template<class T>
struct has_get_t1_impl<T, void_t<typename T::template get_t<>>> : std::true_type {};

template<class T>
struct has_get_t2_impl<T, void_t<typename T::template get_t<std::integral_constant<std::size_t, 0>>>> : std::true_type {};

template<class T>
struct has_get_t_impl<T, std::enable_if_t<has_get_t1_impl<T>::type::value || has_get_t2_impl<T>::type::value>> : std::true_type {};

template<class T>
struct is_structoid_impl<T, std::enable_if_t<has_construct<T>::value>> : std::true_type {};

template<class T>
struct is_struct_impl<T, std::enable_if_t<is_structoid<T>::value && has_get_t<T>::value>> : std::true_type {};

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_IS_STRUCT_HPP

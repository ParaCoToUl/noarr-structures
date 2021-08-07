#ifndef NOARR_STRUCTURES_IS_STRUCT_HPP
#define NOARR_STRUCTURES_IS_STRUCT_HPP

#include "std_ext.hpp"
#include "pipes.hpp"

// TODO: is_struct and is_structoid should be more strict

namespace noarr {

namespace helpers {

template<typename T, typename = void>
struct is_struct_impl {
	using type = std::false_type;
};

template<typename T, typename = void>
struct is_structoid_impl {
	using type = std::false_type;
};

template<typename T, typename = void>
struct has_construct_impl {
	using type = std::false_type;
};

template<typename T, typename = void>
struct has_get_t_impl {
	using type = std::false_type;
};

template<typename T, typename = void>
struct has_get_t1_impl {
	using type = std::false_type;
};

template<typename T, typename = void>
struct has_get_t2_impl {
	using type = std::false_type;
};

template<typename T, typename = void>
struct has_size_impl;

} // namespace helpers

/**
 * @brief returns whether the type `T` has a suitable `construct` method
 * 
 * @tparam T: the input type
 */
template<typename T>
using has_construct = typename helpers::has_construct_impl<T>::type;

/**
 * @brief returns whether the type `T` has a `get_t` member typedef
 * 
 * @tparam T: the input type
 */
template<typename T>
using has_get_t = typename helpers::has_get_t_impl<T>::type;

/**
 * @brief returns whether the type `T` meets the criteria for structures
 * 
 * @tparam T: the input type
 */
template<typename T>
using is_struct = typename helpers::is_struct_impl<T>::type;

/**
 * @brief returns whether the type `T` meets the criteria for structoids
 * 
 * @tparam T: the input type
 */
template<typename T>
using is_structoid = typename helpers::is_structoid_impl<T>::type;

namespace helpers {

template<typename T>
struct has_construct_impl<T, void_t<decltype(construct(std::declval<T>(), std::declval<typename sub_structures<T>::value_type>()))>> {
	using type = std::true_type;
};

template<typename T>
struct has_get_t1_impl<T, void_t<typename T::template get_t<>>> {
	using type = std::true_type;
};

template<typename T>
struct has_get_t2_impl<T, void_t<typename T::template get_t<std::integral_constant<std::size_t, 0>>>> {
	using type = std::true_type;
};

template<typename T>
struct has_get_t_impl<T, std::enable_if_t<has_get_t1_impl<T>::type::value || has_get_t2_impl<T>::type::value>> {
	using type = std::true_type;
};

template<typename T>
struct is_structoid_impl<T, std::enable_if_t<has_construct<T>::value>> {
	using type = std::true_type;
};

template<typename T>
struct is_struct_impl<T, std::enable_if_t<is_structoid<T>::value && has_get_t<T>::value>> {
	using type = std::true_type;
};

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_IS_STRUCT_HPP
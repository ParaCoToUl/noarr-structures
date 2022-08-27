#ifndef NOARR_STRUCTURES_STRUCT_TRAITS_HPP
#define NOARR_STRUCTURES_STRUCT_TRAITS_HPP

#include "type.hpp"

namespace noarr {

template<class T>
struct type_is_point : std::false_type {
	static_assert(is_struct_type<T>(), "type_is_point is only applicable to struct types (type.hpp)");
};
template<class ValueType>
struct type_is_point<scalar_type<ValueType>> : std::true_type {};

/**
 * @brief returns whether the structure is a point (a structure with no dimensions, or with all dimensions being fixed)
 * 
 * @tparam T: the structure
 */
template<class T>
struct is_point : type_is_point<typename T::struct_type> {};



template<class T>
struct type_is_cube : std::false_type {
	static_assert(is_struct_type<T>(), "type_is_cube is only applicable to struct types (type.hpp)");
};
template<class ValueType>
struct type_is_cube<scalar_type<ValueType>> : std::true_type {};
template<char Dim, class ArgLength, class RetType>
struct type_is_cube<function_type<Dim, ArgLength, RetType>> : std::integral_constant<bool, ArgLength::is_known && type_is_cube<RetType>()> {};

/**
 * @brief returns whether a structure is a cube (its dimension and dimension of its substructures, recursively, are all dynamic)
 * 
 * @tparam T: the structure
 */
template<class T>
struct is_cube : type_is_cube<typename T::struct_type> {};



namespace helpers {

template<class T, class State>
struct scalar_t_impl;
template<char Dim, class ArgLength, class RetType, class State>
struct scalar_t_impl<function_type<Dim, ArgLength, RetType>, State> {
	static_assert(State::template contains<index_in<Dim>>, "Not all dimensions are fixed");
	using type = typename scalar_t_impl<RetType, typename State::remove_t<index_in<Dim>, length_in<Dim>>>::type;
};
template<char Dim, class... RetTypes, class State>
struct scalar_t_impl<dep_function_type<Dim, RetTypes...>, State> {
	static_assert(State::template contains<index_in<Dim>>, "Not all dimensions are fixed");
	static_assert(State::template get_t<index_in<Dim>>::value || true, "Tuple index must be set statically, add _idx to the index (e.g. replace 42 with 42_idx)");
	using type = typename scalar_t_impl<typename dep_function_type<Dim, RetTypes...>::ret_type<State::template get_t<index_in<Dim>>::value>, typename State::template remove_t<index_in<Dim>, length_in<Dim>>>::type;
};
template<class ValueType, class State>
struct scalar_t_impl<scalar_type<ValueType>, State> {
	static_assert(State::is_empty, "Superfluous parameters passed in the state");
	using type = ValueType;
};

} // namespace helpers

/**
 * @brief returns the type of the value described by a `scalar<...>`
 * 
 * @tparam T: the `scalar<...>`
 */
template<class T, class State = state<>>
using scalar_t = typename helpers::scalar_t_impl<T, State>::type;

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_TRAITS_HPP

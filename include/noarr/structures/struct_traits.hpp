#ifndef NOARR_STRUCTURES_STRUCT_TRAITS_HPP
#define NOARR_STRUCTURES_STRUCT_TRAITS_HPP

#include "signature.hpp"

namespace noarr {

template<class T>
struct sig_is_point : std::false_type {
	static_assert(is_signature<T>(), "sig_is_point is only applicable to struct signatures");
};
template<class ValueType>
struct sig_is_point<scalar_sig<ValueType>> : std::true_type {};

/**
 * @brief returns whether the structure is a point (a structure with no dimensions, or with all dimensions being fixed)
 * 
 * @tparam T: the structure
 */
template<class T>
struct is_point : sig_is_point<typename T::signature> {};



template<class T>
struct sig_is_cube : std::false_type {
	static_assert(is_signature<T>(), "sig_is_cube is only applicable to struct signatures");
};
template<class ValueType>
struct sig_is_cube<scalar_sig<ValueType>> : std::true_type {};
template<char Dim, class ArgLength, class RetSig>
struct sig_is_cube<function_sig<Dim, ArgLength, RetSig>> : std::integral_constant<bool, ArgLength::is_known && sig_is_cube<RetSig>()> {};

/**
 * @brief returns whether a structure is a cube (its dimension and dimension of its substructures, recursively, are all dynamic)
 * 
 * @tparam T: the structure
 */
template<class T>
struct is_cube : sig_is_cube<typename T::signature> {};



namespace helpers {

template<class T, class State>
struct scalar_t_impl;
template<char Dim, class ArgLength, class RetSig, class State>
struct scalar_t_impl<function_sig<Dim, ArgLength, RetSig>, State> {
	static_assert(State::template contains<index_in<Dim>>, "Not all dimensions are fixed");
	using type = typename scalar_t_impl<RetSig, typename State::remove_t<index_in<Dim>, length_in<Dim>>>::type;
};
template<char Dim, class... RetSigs, class State>
struct scalar_t_impl<dep_function_sig<Dim, RetSigs...>, State> {
	static_assert(State::template contains<index_in<Dim>>, "Not all dimensions are fixed");
	static_assert(State::template get_t<index_in<Dim>>::value || true, "Tuple index must be set statically, add _idx to the index (e.g. replace 42 with 42_idx)");
	using type = typename scalar_t_impl<typename dep_function_sig<Dim, RetSigs...>::ret_sig<State::template get_t<index_in<Dim>>::value>, typename State::template remove_t<index_in<Dim>, length_in<Dim>>>::type;
};
template<class ValueType, class State>
struct scalar_t_impl<scalar_sig<ValueType>, State> {
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

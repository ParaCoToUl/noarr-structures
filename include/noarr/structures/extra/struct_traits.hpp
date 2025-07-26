#ifndef NOARR_STRUCTURES_STRUCT_TRAITS_HPP
#define NOARR_STRUCTURES_STRUCT_TRAITS_HPP

#include <type_traits>

#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/utility.hpp"

namespace noarr {

template<class T>
struct sig_is_point : std::false_type {
	static_assert(is_signature_v<T>, "sig_is_point is only applicable to struct signatures");
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
	static_assert(is_signature_v<T>, "sig_is_cube is only applicable to struct signatures");
};

template<class ValueType>
struct sig_is_cube<scalar_sig<ValueType>> : std::true_type {};

template<IsDim auto Dim, class ArgLength, class RetSig>
struct sig_is_cube<function_sig<Dim, ArgLength, RetSig>> : std::integral_constant<bool, sig_is_cube<RetSig>::value> {};

/**
 * @brief returns whether a structure is a cube (its dimension and dimension of its substructures, recursively, are all
 * dynamic)
 *
 * @tparam T: the structure
 */
template<class T>
struct is_cube {
	static constexpr bool value = sig_is_cube<typename T::signature>::value && requires {
		T::template has_size<state<>>();
		typename std::enable_if_t<T::template has_size<state<>>()>;
	};
};

template<class T, IsState State>
struct sig_get_scalar;

template<IsDim auto Dim, class ArgLength, class RetSig, IsState State>
struct sig_get_scalar<function_sig<Dim, ArgLength, RetSig>, State> {
	using type = typename sig_get_scalar<RetSig, state_remove_t<State, index_in<Dim>, length_in<Dim>>>::type;
};

template<IsDim auto Dim, class... RetSigs, IsState State>
requires (state_contains<State, index_in<Dim>>)
struct sig_get_scalar<dep_function_sig<Dim, RetSigs...>, State> {
	static_assert(
		requires { state_get_t<State, index_in<Dim>>::value; },
		"Tuple index must be set statically, wrap it in lit<> (e.g. replace 42 with lit<42>)");
	using type = typename sig_get_scalar<
		typename dep_function_sig<Dim, RetSigs...>::template ret_sig<state_get_t<State, index_in<Dim>>::value>,
		state_remove_t<State, index_in<Dim>, length_in<Dim>>>::type;
};

template<class ValueType, IsState State>
struct sig_get_scalar<scalar_sig<ValueType>, State> {
	using type = ValueType;
};

/**
 * @brief returns the type of the value described by a `scalar<...>`
 *
 * @tparam T: the `scalar<...>`
 */
template<class T, IsState State = state<>>
using scalar_t = typename sig_get_scalar<typename std::remove_cvref_t<T>::signature, State>::type;

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_TRAITS_HPP

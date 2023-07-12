#ifndef NOARR_STRUCTURES_STRUCT_TRAITS_HPP
#define NOARR_STRUCTURES_STRUCT_TRAITS_HPP

#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/utility.hpp"

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
template<IsDim auto Dim, class ArgLength, class RetSig>
struct sig_is_cube<function_sig<Dim, ArgLength, RetSig>> : std::integral_constant<bool, ArgLength::is_known && sig_is_cube<RetSig>()> {};

/**
 * @brief returns whether a structure is a cube (its dimension and dimension of its substructures, recursively, are all dynamic)
 * 
 * @tparam T: the structure
 */
template<class T>
struct is_cube : sig_is_cube<typename T::signature> {};



template<class T, IsState State>
struct sig_get_scalar;
template<IsDim auto Dim, class ArgLength, class RetSig, IsState State>
struct sig_get_scalar<function_sig<Dim, ArgLength, RetSig>, State> {
	using type = typename sig_get_scalar<RetSig, state_remove_t<State, index_in<Dim>, length_in<Dim>>>::type;
};
template<IsDim auto Dim, class... RetSigs, IsState State>
struct sig_get_scalar<dep_function_sig<Dim, RetSigs...>, State> {
	static_assert(State::template contains<index_in<Dim>>, "Not all tuple dimensions are fixed");
	static_assert(state_get_t<State, index_in<Dim>>::value || true, "Tuple index must be set statically, wrap it in lit<> (e.g. replace 42 with lit<42>)");
	using type = typename sig_get_scalar<typename dep_function_sig<Dim, RetSigs...>::template ret_sig<state_get_t<State, index_in<Dim>>::value>, state_remove_t<State, index_in<Dim>, length_in<Dim>>>::type;
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
using scalar_t = typename sig_get_scalar<typename T::signature, State>::type;

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_TRAITS_HPP

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
template<char Dim, class ArgLength, class RetSig>
struct sig_is_cube<function_sig<Dim, ArgLength, RetSig>> : std::integral_constant<bool, ArgLength::is_known && sig_is_cube<RetSig>()> {};

/**
 * @brief returns whether a structure is a cube (its dimension and dimension of its substructures, recursively, are all dynamic)
 * 
 * @tparam T: the structure
 */
template<class T>
struct is_cube : sig_is_cube<typename T::signature> {};



template<class T, class State, bool Checked>
struct sig_get_scalar;
template<char Dim, class ArgLength, class RetSig, class State, bool Checked>
struct sig_get_scalar<function_sig<Dim, ArgLength, RetSig>, State, Checked> {
	static_assert(State::template contains<index_in<Dim>> || !Checked, "Not all dimensions are fixed");
	using type = typename sig_get_scalar<RetSig, state_remove_t<State, index_in<Dim>, length_in<Dim>>, Checked>::type;
};
template<char Dim, class... RetSigs, class State, bool Checked>
struct sig_get_scalar<dep_function_sig<Dim, RetSigs...>, State, Checked> {
	static_assert(State::template contains<index_in<Dim>>, "Not all dimensions are fixed");
	static_assert(state_get_t<State, index_in<Dim>>::value || true, "Tuple index must be set statically, wrap it in idx<> (e.g. replace 42 with idx<42>)");
	using type = typename sig_get_scalar<typename dep_function_sig<Dim, RetSigs...>::template ret_sig<state_get_t<State, index_in<Dim>>::value>, state_remove_t<State, index_in<Dim>, length_in<Dim>>, Checked>::type;
};
template<class ValueType, class State, bool Checked>
struct sig_get_scalar<scalar_sig<ValueType>, State, Checked> {
	static_assert(State::is_empty || !Checked, "Superfluous parameters passed in the state");
	using type = ValueType;
};

/**
 * @brief returns the type of the value described by a `scalar<...>`
 * 
 * @tparam T: the `scalar<...>`
 */
template<class T, class State = state<>>
using scalar_t = typename sig_get_scalar<typename T::signature, State, true>::type;

template<class T, class State = state<>>
using unchecked_scalar_t = typename sig_get_scalar<typename T::signature, State, false>::type;

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_TRAITS_HPP

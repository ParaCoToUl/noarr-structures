#ifndef NOARR_STRUCTURES_SIG_UTILS_HPP
#define NOARR_STRUCTURES_SIG_UTILS_HPP

#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/utility.hpp"

namespace noarr {

namespace helpers {

template<char QDim, class State, class Signature>
struct sig_find_dim_impl;

template<char QDim, class State, class ArgLength, class RetSig>
struct sig_find_dim_impl<QDim, State, function_sig<QDim, ArgLength, RetSig>> {
	using type = function_sig<QDim, ArgLength, RetSig>;
};
template<char QDim, class State, class... RetSigs>
struct sig_find_dim_impl<QDim, State, dep_function_sig<QDim, RetSigs...>> {
	using type = dep_function_sig<QDim, RetSigs...>;
};
template<char QDim, class State, char Dim, class ArgLength, class RetSig>
struct sig_find_dim_impl<QDim, State, function_sig<Dim, ArgLength, RetSig>> {
	using type = typename sig_find_dim_impl<QDim, State, RetSig>::type;
};
template<char QDim, class State, char Dim, class... RetSigs>
struct sig_find_dim_impl<QDim, State, dep_function_sig<Dim, RetSigs...>> {
	static_assert(State::template contains<index_in<Dim>>, "Cannot extract dimension from within tuple");
	static constexpr std::size_t idx = state_get_t<State, index_in<Dim>>::value;
	using ret_sig = typename dep_function_sig<Dim, RetSigs...>::template ret_sig<idx>;
	using type = typename sig_find_dim_impl<QDim, State, ret_sig>::type;
};
template<char QDim, class State, class ValueType>
struct sig_find_dim_impl<QDim, State, scalar_sig<ValueType>> {
	static_assert(value_always_false<QDim>, "The structure does not have a dimension of this name");
};

} // namespace helpers

template<char QDim, class State, class Signature>
using sig_find_dim = typename helpers::sig_find_dim_impl<QDim, State, Signature>::type;

} // namespace noarr

#endif // NOARR_STRUCTURES_SIG_UTILS_HPP

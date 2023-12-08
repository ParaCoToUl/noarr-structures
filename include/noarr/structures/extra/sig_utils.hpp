#ifndef NOARR_STRUCTURES_SIG_UTILS_HPP
#define NOARR_STRUCTURES_SIG_UTILS_HPP

#include <cstddef>

#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/utility.hpp"

namespace noarr {

namespace helpers {

template<IsDim auto QDim, IsState State, class Signature>
struct sig_find_dim_impl;

template<IsDim auto QDim, IsState State, IsDim auto Dim, class ArgLength, class RetSig> requires (Dim != QDim)
struct sig_find_dim_impl<QDim, State, function_sig<Dim, ArgLength, RetSig>> {
	using type = typename sig_find_dim_impl<QDim, State, RetSig>::type;
};
template<IsDim auto QDim, IsState State, class ArgLength, class RetSig>
struct sig_find_dim_impl<QDim, State, function_sig<QDim, ArgLength, RetSig>> {
	using type = function_sig<QDim, ArgLength, RetSig>;
};
template<IsDim auto QDim, IsState State, IsDim auto Dim, class ...RetSigs> requires (Dim != QDim)
struct sig_find_dim_impl<QDim, State, dep_function_sig<Dim, RetSigs...>> {
	static_assert(State::template contains<index_in<Dim>>, "Cannot extract dimension from within tuple");
	static constexpr std::size_t idx = state_get_t<State, index_in<Dim>>::value;
	using ret_sig = typename dep_function_sig<Dim, RetSigs...>::template ret_sig<idx>;
	using type = typename sig_find_dim_impl<QDim, State, ret_sig>::type;
};
template<IsDim auto QDim, IsState State, class ...RetSigs>
struct sig_find_dim_impl<QDim, State, dep_function_sig<QDim, RetSigs...>> {
	using type = dep_function_sig<QDim, RetSigs...>;
};
template<IsDim auto QDim, IsState State, class ValueType>
struct sig_find_dim_impl<QDim, State, scalar_sig<ValueType>> {
	static_assert(value_always_false<QDim>, "The structure does not have a dimension of this name");
};

template<class Signature>
struct sig_dim_tree_impl;

template<IsDim auto Dim, class ArgLength, class RetSig>
struct sig_dim_tree_impl<function_sig<Dim, ArgLength, RetSig>> {
	using type = dim_tree<Dim, typename sig_dim_tree_impl<RetSig>::type>;
};

template<IsDim auto Dim, class ...RetSigs>
struct sig_dim_tree_impl<dep_function_sig<Dim, RetSigs...>> {
	using type = dim_tree<Dim, typename sig_dim_tree_impl<RetSigs>::type...>;
};

template<class ValueType>
struct sig_dim_tree_impl<scalar_sig<ValueType>> {
	using type = dim_sequence<>;
};

} // namespace helpers

template<IsDim auto QDim, IsState State, class Signature>
using sig_find_dim = typename helpers::sig_find_dim_impl<QDim, State, Signature>::type;

template<class Signature>
using sig_dim_tree = typename helpers::sig_dim_tree_impl<Signature>::type;

} // namespace noarr

#endif // NOARR_STRUCTURES_SIG_UTILS_HPP

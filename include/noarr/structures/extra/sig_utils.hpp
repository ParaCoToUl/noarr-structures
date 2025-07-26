#ifndef NOARR_STRUCTURES_SIG_UTILS_HPP
#define NOARR_STRUCTURES_SIG_UTILS_HPP

#include <cstddef>

#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/utility.hpp"

namespace noarr {

namespace helpers {

template<auto QDim, IsState State, class Signature>
requires IsDim<decltype(QDim)>
struct sig_find_dim_impl;

template<auto QDim, IsState State, auto Dim, class ArgLength, class RetSig>
requires (Dim != QDim) && IsDim<decltype(Dim)> && IsDim<decltype(QDim)>
struct sig_find_dim_impl<QDim, State, function_sig<Dim, ArgLength, RetSig>> {
	using type = typename sig_find_dim_impl<QDim, State, RetSig>::type;
};

template<auto QDim, IsState State, class ArgLength, class RetSig>
requires IsDim<decltype(QDim)>
struct sig_find_dim_impl<QDim, State, function_sig<QDim, ArgLength, RetSig>> {
	using type = function_sig<QDim, ArgLength, RetSig>;
};

template<auto QDim, IsState State, auto Dim, class... RetSigs>
requires (Dim != QDim) && IsDim<decltype(Dim)> && IsDim<decltype(QDim)>
struct sig_find_dim_impl<QDim, State, dep_function_sig<Dim, RetSigs...>> {
	static_assert(state_contains<State, index_in<Dim>>, "Cannot extract dimension from within tuple");
	static constexpr std::size_t idx = state_get_t<State, index_in<Dim>>::value;
	using ret_sig = typename dep_function_sig<Dim, RetSigs...>::template ret_sig<idx>;
	using type = typename sig_find_dim_impl<QDim, State, ret_sig>::type;
};

template<auto QDim, IsState State, class... RetSigs>
requires IsDim<decltype(QDim)>
struct sig_find_dim_impl<QDim, State, dep_function_sig<QDim, RetSigs...>> {
	using type = dep_function_sig<QDim, RetSigs...>;
};

template<auto QDim, IsState State, class ValueType>
requires IsDim<decltype(QDim)>
struct sig_find_dim_impl<QDim, State, scalar_sig<ValueType>> {
	static_assert(value_always_false<QDim>, "The structure does not have a dimension of this name");
};

template<class Signature>
struct sig_dim_tree_impl;

template<auto Dim, class ArgLength, class RetSig>
requires IsDim<decltype(Dim)>
struct sig_dim_tree_impl<function_sig<Dim, ArgLength, RetSig>> {
	using type = dim_tree<Dim, typename sig_dim_tree_impl<RetSig>::type>;
};

template<auto Dim, class... RetSigs>
requires IsDim<decltype(Dim)>
struct sig_dim_tree_impl<dep_function_sig<Dim, RetSigs...>> {
	using type = dim_tree<Dim, typename sig_dim_tree_impl<RetSigs>::type...>;
};

template<class ValueType>
struct sig_dim_tree_impl<scalar_sig<ValueType>> {
	using type = dim_sequence<>;
};

template<class Signature>
struct sig_dim_seq_impl;

template<auto Dim, class ArgLength, class RetSig>
requires IsDim<decltype(Dim)>
struct sig_dim_seq_impl<function_sig<Dim, ArgLength, RetSig>> {
	using ret_type = typename sig_dim_seq_impl<RetSig>::type;
	using type = typename ret_type::template push_back<Dim>;
};

template<auto Dim, class RetSigs>
requires IsDim<decltype(Dim)>
struct sig_dim_seq_impl<dep_function_sig<Dim, RetSigs>> {
	using ret_type = typename sig_dim_seq_impl<RetSigs>::type;
	using type = typename ret_type::template push_back<Dim>;
};

template<auto Dim, class RetSig, class... RetSigs>
requires IsDim<decltype(Dim)>
struct sig_dim_seq_impl<dep_function_sig<Dim, RetSig, RetSigs...>> {
	static_assert(always_false<sig_dim_seq_impl>, "Dependent function signatures are not supported for this operation");
	using ret_type = typename sig_dim_seq_impl<RetSig>::type;
	using type = typename ret_type::template push_back<Dim>;
};

template<class ValueType>
struct sig_dim_seq_impl<scalar_sig<ValueType>> {
	using type = dim_sequence<>;
};

} // namespace helpers

template<auto QDim, IsState State, class Signature>
requires IsDim<decltype(QDim)>
using sig_find_dim = typename helpers::sig_find_dim_impl<QDim, State, Signature>::type;

template<class Signature>
using sig_dim_tree = typename helpers::sig_dim_tree_impl<Signature>::type;

template<class Signature>
using sig_dim_seq = typename helpers::sig_dim_seq_impl<Signature>::type;

template<class OriginalSig>
struct sig_remove_first {
	using type = typename OriginalSig::ret_sig;
};

template<auto Dim, class ArgLength, class RetSig>
struct sig_remove_first<function_sig<Dim, ArgLength, RetSig>> {
	using type = RetSig;
};

template<auto Dim, class RetSigs>
struct sig_remove_first<dep_function_sig<Dim, RetSigs>> {
	using type = RetSigs;
};

template<auto Dim, class RetSig, class... RetSigs>
struct sig_remove_first<dep_function_sig<Dim, RetSig, RetSigs...>> {
	static_assert(always_false<sig_remove_first>, "Dependent function signatures are not supported for this operation");
	using type = RetSig;
};

template<class Sig>
struct in_signature {
	using signature = Sig;
	using value_type = bool;

	template<IsDim auto Dim>
	static constexpr bool value = Sig::template any_accept<Dim>;
};

} // namespace noarr

#endif // NOARR_STRUCTURES_SIG_UTILS_HPP

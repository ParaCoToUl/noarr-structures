#ifndef NOARR_STRUCTURES_SIGNATURE_HPP
#define NOARR_STRUCTURES_SIGNATURE_HPP

#include "contain.hpp"

namespace noarr {

struct unknown_arg_length {
	static constexpr bool valid_arg_length = true;
	static constexpr bool is_known = false;
	static constexpr bool is_static = false;
};
struct dynamic_arg_length {
	static constexpr bool valid_arg_length = true;
	static constexpr bool is_known = true;
	static constexpr bool is_static = false;
};
template<std::size_t L>
struct static_arg_length {
	static constexpr bool valid_arg_length = true;
	static constexpr bool is_known = true;
	static constexpr bool is_static = true;
	static constexpr std::size_t value = L;
};

namespace helpers {

template<class T>
struct arg_length_from;
template<>
struct arg_length_from<std::size_t> { using type = dynamic_arg_length; };
template<std::size_t L>
struct arg_length_from<std::integral_constant<std::size_t, L>> { using type = static_arg_length<L>; };

} // namespace helpers

template<class T>
using arg_length_from_t = typename helpers::arg_length_from<T>::type;

template<IsDim auto Dim, class ArgLength, class RetSig>
struct function_sig {
	static_assert(ArgLength::valid_arg_length);
	function_sig() = delete;

	static constexpr auto dim = Dim;
	using arg_length = ArgLength;
	using ret_sig = RetSig;

private:
	template<bool Match, template<class Original> class Replacement, auto... QDims>
	struct replace_inner;
	template<template<class Original> class Replacement, auto... QDims>
	struct replace_inner<true, Replacement, QDims...> { using type = typename Replacement<function_sig>::type; };
	template<template<class Original> class Replacement, auto... QDims>
	struct replace_inner<false, Replacement, QDims...> { using type = function_sig<Dim, ArgLength, typename RetSig::template replace<Replacement, QDims...>>; };
public:
	template<template<class Original> class Replacement, auto... QDims> requires (... && IsDim<decltype(QDims)>)
	using replace = typename replace_inner<((QDims == Dim) || ...), Replacement, QDims...>::type;

	template<IsDim auto QDim>
	static constexpr bool all_accept = (Dim == QDim || RetSig::template all_accept<QDim>);
	template<IsDim auto QDim>
	static constexpr bool any_accept = (Dim == QDim || RetSig::template any_accept<QDim>);

	static constexpr bool dependent = false;
};

template<IsDim auto Dim, class... RetSigs>
struct dep_function_sig {
	dep_function_sig() = delete;

	static constexpr auto dim = Dim;
	using ret_sig_tuple = std::tuple<RetSigs...>;
	template<std::size_t N>
	using ret_sig = std::tuple_element_t<N, ret_sig_tuple>;

private:
	template<bool Match, template<class Original> class Replacement, auto... QDims>
	struct replace_inner;
	template<template<class Original> class Replacement, auto... QDims>
	struct replace_inner<true, Replacement, QDims...> { using type = typename Replacement<dep_function_sig>::type; };
	template<template<class Original> class Replacement, auto... QDims>
	struct replace_inner<false, Replacement, QDims...> { using type = dep_function_sig<Dim, typename RetSigs::template replace<Replacement, QDims...>...>; };
public:
	template<template<class Original> class Replacement, auto... QDims> requires (... && IsDim<decltype(QDims)>)
	using replace = typename replace_inner<((QDims == Dim) || ...), Replacement, QDims...>::type;

	template<IsDim auto QDim>
	static constexpr bool all_accept = (Dim == QDim || (RetSigs::template all_accept<QDim> && ...));
	template<IsDim auto QDim>
	static constexpr bool any_accept = (Dim == QDim || (RetSigs::template any_accept<QDim> || ...));

	static constexpr bool dependent = true;
};

template<class ValueType>
struct scalar_sig {
	scalar_sig() = delete;

	template<template<class Original> class Replacement, auto... QDims> requires (... && IsDim<decltype(QDims)>)
	using replace = scalar_sig;

	template<IsDim auto QDim>
	static constexpr bool all_accept = false;
	template<IsDim auto QDim>
	static constexpr bool any_accept = false;
};

template<class T>
struct is_function_sig : std::false_type {};
template<IsDim auto Dim, class ArgLength, class RetSig>
struct is_function_sig<function_sig<Dim, ArgLength, RetSig>> : std::true_type {};
template<IsDim auto Dim, class... RetSigs>
struct is_function_sig<dep_function_sig<Dim, RetSigs...>> : std::true_type {};

template<class T>
static constexpr bool is_function_sig_v = is_function_sig<T>::value;

template<class T>
concept IsFunctionSig = is_function_sig_v<T>;

template<class T>
struct is_ground_sig : std::false_type {};
template<class ValueType>
struct is_ground_sig<scalar_sig<ValueType>> : std::true_type {};

template<class T>
static constexpr bool is_ground_sig_v = is_ground_sig<T>::value;

template<class T>
concept IsGroundSig = is_ground_sig_v<T>;

template<class T>
struct is_signature : std::false_type {};
template<IsDim auto Dim, class ArgLength, class RetSig>
struct is_signature<function_sig<Dim, ArgLength, RetSig>> : std::true_type {};
template<IsDim auto Dim, class... RetSigs>
struct is_signature<dep_function_sig<Dim, RetSigs...>> : std::true_type {};
template<class ValueType>
struct is_signature<scalar_sig<ValueType>> : std::true_type {};

template<class T>
static constexpr bool is_signature_v = is_signature<T>::value;

template<class T>
concept IsSignature = is_signature_v<T>;

} // namespace noarr

#endif // NOARR_STRUCTURES_SIGNATURE_HPP

#ifndef NOARR_STRUCTURES_TYPES_HPP
#define NOARR_STRUCTURES_TYPES_HPP

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

template<char Dim, class ArgLength, class RetType>
struct function_type {
	static_assert(ArgLength::valid_arg_length);
	function_type() = delete;

	static constexpr char dim = Dim;
	using arg_length = ArgLength;
	using ret_type = RetType;

private:
	template<bool Match, template<class Original> class Replacement, char... QDims>
	struct replace_inner;
	template<template<class Original> class Replacement, char... QDims>
	struct replace_inner<true, Replacement, QDims...> { using type = typename Replacement<function_type>::type; };
	template<template<class Original> class Replacement, char... QDims>
	struct replace_inner<false, Replacement, QDims...> { using type = function_type<Dim, ArgLength, typename RetType::replace<Replacement, QDims...>>; };
public:
	template<template<class Original> class Replacement, char... QDims>
	using replace = typename replace_inner<((QDims == Dim) || ...), Replacement, QDims...>::type;

	template<char QDim>
	static constexpr bool all_accept = (Dim == QDim || RetType::template all_accept<QDim>);
	template<char QDim>
	static constexpr bool any_accept = (Dim == QDim || RetType::template any_accept<QDim>);

	static constexpr bool dependent = false;
};

template<char Dim, class... RetTypes>
struct dep_function_type {
	dep_function_type() = delete;

	static constexpr char dim = Dim;
	using ret_type_tuple = std::tuple<RetTypes...>;
	template<std::size_t N>
	using ret_type = typename std::tuple_element<N, ret_type_tuple>::type;

private:
	template<bool Match, template<class Original> class Replacement, char... QDims>
	struct replace_inner;
	template<template<class Original> class Replacement, char... QDims>
	struct replace_inner<true, Replacement, QDims...> { using type = typename Replacement<dep_function_type>::type; };
	template<template<class Original> class Replacement, char... QDims>
	struct replace_inner<false, Replacement, QDims...> { using type = dep_function_type<Dim, typename RetTypes::replace<Replacement, QDims...>...>; };
public:
	template<template<class Original> class Replacement, char... QDims>
	using replace = typename replace_inner<((QDims == Dim) || ...), Replacement, QDims...>::type;

	template<char QDim>
	static constexpr bool all_accept = (Dim == QDim || (RetTypes::template all_accept<QDim> && ...));
	template<char QDim>
	static constexpr bool any_accept = (Dim == QDim || (RetTypes::template any_accept<QDim> || ...));

	static constexpr bool dependent = true;
};

template<class ValueType>
struct scalar_type {
	scalar_type() = delete;

	template<char QDim>
	static constexpr bool all_accept = false;
	template<char QDim>
	static constexpr bool any_accept = false;
};

template<class T>
struct is_struct_type : std::false_type {};
template<char Dim, class ArgLength, class RetType>
struct is_struct_type<function_type<Dim, ArgLength, RetType>> : std::true_type {};
template<char Dim, class... RetTypes>
struct is_struct_type<dep_function_type<Dim, RetTypes...>> : std::true_type {};
template<class ValueType>
struct is_struct_type<scalar_type<ValueType>> : std::true_type {};

} // namespace noarr

#endif // NOARR_STRUCTURES_TYPES_HPP

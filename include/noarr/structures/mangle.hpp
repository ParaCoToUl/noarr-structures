#ifndef NOARR_STRUCTURES_MANGLE_HPP
#define NOARR_STRUCTURES_MANGLE_HPP

#include "utility.hpp"
#include "scalar.hpp"
#include "structs_common.hpp"

namespace noarr {

template<class CharPack>
struct char_seq_to_str;

template<char... C>
struct char_seq_to_str<char_sequence<C...>> {
	static constexpr char c_str[] = {C..., '\0'};
	static constexpr std::size_t length = sizeof...(C);
};

namespace helpers {

template<const char Name[], class Indices, class Params>
struct mangle_desc;

}

/**
 * @brief Returns a textual representation of the type of a structure using `char_sequence`
 * 
 * @tparam T: the structure
 */
template<class T>
using mangle = typename helpers::mangle_desc<T::name, std::make_index_sequence<sizeof(T::name) - 1>, typename T::params>::type;

template<class T>
using mangle_to_str = char_seq_to_str<mangle<T>>;

namespace helpers {

template<class T, T V, class = void>
struct mangle_value_impl;

template<class T, T V>
using mangle_value = typename mangle_value_impl<T, V>::type;

template<class T, T V, class Acc = std::integer_sequence<char>, class = void>
struct mangle_integral;

template<class T, char... Acc, T V>
struct mangle_integral<T, V, char_sequence<Acc...>, std::enable_if_t<(V >= 10)>>
	: mangle_integral<T, V / 10, char_sequence<(char)(V % 10) + '0', Acc...>> {};

template<class T, char... Acc, T V>
struct mangle_integral<T, V, char_sequence<Acc...>, std::enable_if_t<(V < 10 && V >= 0)>> {
	using type = char_sequence<(char)(V % 10) + '0', Acc...>;
};

template<class T, T V>
struct mangle_value_impl<T, V, std::enable_if_t<std::is_integral<T>::value>>
	: mangle_integral<T, V> {};

/**
 * @brief returns a textual representation of a scalar type using `char_sequence`
 * 
 * @tparam T: the scalar type
 */
template<class T, class = void>
struct scalar_name;

template<class T, class>
struct scalar_name {
	static_assert(always_false<T>, "scalar_name<T> has to be implemented");
	using type = void;
};

template<class T>
struct scalar_name<T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>> {
	using type = integer_sequence_concat<char_sequence<'i', 'n', 't'>, mangle_value<int, 8 * sizeof(T)>, char_sequence<'_', 't'>>;
};

template<class T>
struct scalar_name<T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>> {
	using type = integer_sequence_concat<char_sequence<'u', 'i', 'n', 't'>, mangle_value<int, 8 * sizeof(T)>, char_sequence<'_', 't'>>;
};

template<>
struct scalar_name<float> {
	using type = char_sequence<'f', 'l', 'o', 'a', 't'>;
};

template<>
struct scalar_name<double> {
	using type = char_sequence<'d', 'o', 'u', 'b', 'l', 'e'>;
};

template<>
struct scalar_name<long double> {
	using type = char_sequence<'l', 'o', 'n', 'g', ' ', 'd', 'o', 'u', 'b', 'l', 'e'>;
};

/**
 * @brief returns a textual representation of a template parameter description using `char_sequence`
 * 
 * @tparam T: one of the arguments to struct_params
 */
template<class T>
struct mangle_param;

template<class T>
struct mangle_param<structure_param<T>> { using type = integer_sequence_concat<mangle<T>>; };

template<class T>
struct mangle_param<type_param<T>> { using type = integer_sequence_concat<typename scalar_name<T>::type>; };

template<class T, T V>
struct mangle_param<value_param<T, V>> { using type = integer_sequence_concat<char_sequence<'('>, typename scalar_name<T>::type, char_sequence<')'>, mangle_value<T, V>>; };

template<char Dim>
struct mangle_param<dim_param<Dim>> { using type = integer_sequence_concat<char_sequence<'\'', Dim, '\''>>; };

template<const char Name[], std::size_t... Indices, class... Params>
struct mangle_desc<Name, std::index_sequence<Indices...>, struct_params<Params...>> {
	using type = integer_sequence_concat<
		char_sequence<Name[Indices]..., '<'>,
		integer_sequence_concat_sep<char_sequence<','>, typename mangle_param<Params>::type...>,
		char_sequence<'>'>>;
};

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_MANGLE_HPP

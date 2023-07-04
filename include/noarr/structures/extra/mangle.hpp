#ifndef NOARR_STRUCTURES_MANGLE_HPP
#define NOARR_STRUCTURES_MANGLE_HPP

#include "../base/contain.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

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

template<class T, T V>
struct mangle_value_impl;

template<class T, T V>
using mangle_value = typename mangle_value_impl<T, V>::type;

template<class T, T V, class Acc = std::integer_sequence<char>>
struct mangle_integral;

template<class T, char... Acc, T V> requires (V >= 10)
struct mangle_integral<T, V, char_sequence<Acc...>>
	: mangle_integral<T, V / 10, char_sequence<(char)(V % 10) + '0', Acc...>> {};

template<class T, char... Acc, T V> requires (V < 10 && V >= 0)
struct mangle_integral<T, V, char_sequence<Acc...>> {
	using type = char_sequence<(char)(V % 10) + '0', Acc...>;
};

template<class T, T V> requires std::is_integral<T>::value
struct mangle_value_impl<T, V>
	: mangle_integral<T, V> {};

/**
 * @brief returns a textual representation of a scalar type using `char_sequence`
 * 
 * @tparam T: the scalar type
 */
template<class T>
struct scalar_name;

template<class T> requires (std::is_integral<T>::value && std::is_signed<T>::value)
struct scalar_name<T> {
	using type = integer_sequence_concat<char_sequence<'i', 'n', 't'>, mangle_value<int, 8 * sizeof(T)>, char_sequence<'_', 't'>>;
};

template<class T> requires (std::is_integral<T>::value && std::is_unsigned<T>::value)
struct scalar_name<T> {
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

template<std::size_t L>
struct scalar_name<std::integral_constant<std::size_t, L>> {
	using type = integer_sequence_concat<char_sequence<'s', 't', 'd', ':', ':', 'i', 'n', 't', 'e', 'g', 'r', 'a', 'l', '_', 'c', 'o', 'n', 's', 't', 'a', 'n', 't', '<'>, typename scalar_name<std::size_t>::type, char_sequence<','>, mangle_value<int, L>, char_sequence<'>'>>;
};

template<std::size_t L>
struct scalar_name<lit_t<L>> {
	using type = integer_sequence_concat<char_sequence<'l', 'i', 't', '_', 't', '<'>, mangle_value<int, L>, char_sequence<'>'>>;
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

struct mangle_expr_helpers {
	template<class... Ts>
	static inline std::index_sequence_for<Ts...> get_contain_indices(const contain<Ts...> &) noexcept; // undefined, use in decltype

	template<class String, class T, std::size_t... Indices>
	static constexpr void append_items(String &out, const T &t, std::index_sequence<Indices...>) {
		(..., (append(out, t.template get<Indices>()), out.push_back(',')));
	}

	template<class String, class U>
	static constexpr void append_int(String &out, bool neg, U u) {
		constexpr auto maxsz = sizeof(U) * 3 + 1; // at most 3 digits per byte + sign
		char buff[maxsz], *ptr = buff + maxsz;
		std::size_t sz = 0;
		if(neg)
			u = -u;
		do {
			*--ptr = '0' + u % 10;
			sz++;
		} while(u /= 10);
		if(neg)
			*--ptr = '-';
		out.append(ptr, sz);
	}

	template<class String, class T, class Indices = decltype(get_contain_indices(std::declval<T>()))> // Indices also used for SFINAE
	static constexpr void append(String &out, const T &t) {
		using type_str = mangle_to_str<T>;
		out.append(type_str::c_str, type_str::length);
		out.push_back('{');
		append_items(out, t, Indices{});
		out.push_back('}');
	}

	template<class String, class T> requires (std::is_integral_v<T>)
	static constexpr void append(String &out, const T &t) {
		using type_str = char_seq_to_str<typename scalar_name<T>::type>;
		out.append(type_str::c_str, type_str::length);
		out.push_back('{');
		append_int(out, (t < 0), std::make_unsigned_t<T>(t));
		out.push_back('}');
	}

	template<class String, std::size_t L>
	static constexpr void append(String &out, const std::integral_constant<std::size_t, L> t) {
		out.append("lit<", 4);
		append_int(out, (t < 0), L);
		out.append(">", 1);
	}
};

} // namespace helpers

template<class String, class T>
constexpr String mangle_expr(const T &t) {
	String out;
	helpers::mangle_expr_helpers::append(out, t);
	return out;
}

} // namespace noarr

#endif // NOARR_STRUCTURES_MANGLE_HPP

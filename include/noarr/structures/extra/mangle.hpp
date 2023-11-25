#ifndef NOARR_STRUCTURES_MANGLE_HPP
#define NOARR_STRUCTURES_MANGLE_HPP

#include <limits>

#include "../base/contain.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

template<class CharPack>
struct char_seq_to_str;

template<char ...C>
struct char_seq_to_str<std::integer_sequence<char, C...>> {
	static constexpr char c_str[] = {C..., '\0'};
	static constexpr std::size_t length = sizeof...(C);
};

namespace helpers {

template<const char Name[], class Indices, class Params>
struct mangle_desc;

}

/**
 * @brief Returns a textual representation of the type of a structure using `std::integer_sequence<char, ...>`
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

template<class T, char ...Acc, T V> requires (V >= 10)
struct mangle_integral<T, V, std::integer_sequence<char, Acc...>>
	: mangle_integral<T, V / 10, std::integer_sequence<char, (char)(V % 10) + '0', Acc...>> {};

template<class T, char ...Acc, T V> requires (V < 10 && V >= 0)
struct mangle_integral<T, V, std::integer_sequence<char, Acc...>> {
	using type = std::integer_sequence<char, (char)(V % 10) + '0', Acc...>;
};

template<class T, T V> requires (std::is_integral_v<T>)
struct mangle_value_impl<T, V>
	: mangle_integral<T, V> {};

/**
 * @brief returns a textual representation of a scalar type using `std::integer_sequence<char, ...>`
 *
 * @tparam T: the scalar type
 */
template<class T>
struct scalar_name;

template<class T> requires (std::is_integral_v<T> && std::is_signed_v<T>)
struct scalar_name<T> {
	using type = integer_sequence_concat<std::integer_sequence<char, 'i', 'n', 't'>, mangle_value<int, 8 * sizeof(T)>, std::integer_sequence<char, '_', 't'>>;
};

template<class T> requires (std::is_integral_v<T> && std::is_unsigned_v<T>)
struct scalar_name<T> {
	using type = integer_sequence_concat<std::integer_sequence<char, 'u', 'i', 'n', 't'>, mangle_value<int, 8 * sizeof(T)>, std::integer_sequence<char, '_', 't'>>;
};

template<>
struct scalar_name<float> {
	using type = std::integer_sequence<char, 'f', 'l', 'o', 'a', 't'>;
};

template<>
struct scalar_name<double> {
	using type = std::integer_sequence<char, 'd', 'o', 'u', 'b', 'l', 'e'>;
};

template<>
struct scalar_name<long double> {
	using type = std::integer_sequence<char, 'l', 'o', 'n', 'g', ' ', 'd', 'o', 'u', 'b', 'l', 'e'>;
};

template<std::size_t L>
struct scalar_name<std::integral_constant<std::size_t, L>> {
	using type = integer_sequence_concat<std::integer_sequence<char, 's', 't', 'd', ':', ':', 'i', 'n', 't', 'e', 'g', 'r', 'a', 'l', '_', 'c', 'o', 'n', 's', 't', 'a', 'n', 't', '<'>, typename scalar_name<std::size_t>::type, std::integer_sequence<char, ','>, mangle_value<int, L>, std::integer_sequence<char, '>'>>;
};

template<std::size_t L>
struct scalar_name<lit_t<L>> {
	using type = integer_sequence_concat<std::integer_sequence<char, 'l', 'i', 't', '_', 't', '<'>, mangle_value<int, L>, std::integer_sequence<char, '>'>>;
};

/**
 * @brief returns a textual representation of a template parameter description using `std::integer_sequence<char,
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
struct mangle_param<value_param<T, V>> { using type = integer_sequence_concat<std::integer_sequence<char, '('>, typename scalar_name<T>::type, std::integer_sequence<char, ')'>, mangle_value<T, V>>; };

template<char Dim>
struct mangle_param<dim_param<Dim>> { using type = integer_sequence_concat<std::integer_sequence<char, '\'', Dim, '\''>>; };

template<const char Name[], std::size_t ...Indices, class ...Params>
struct mangle_desc<Name, std::index_sequence<Indices...>, struct_params<Params...>> {
	using type = integer_sequence_concat<
		std::integer_sequence<char, Name[Indices]..., '<'>,
		integer_sequence_concat_sep<std::integer_sequence<char, ','>, typename mangle_param<Params>::type...>,
		std::integer_sequence<char, '>'>>;
};

struct mangle_expr_helpers {
	template<class ...Ts>
	static inline std::index_sequence_for<Ts...> get_contain_indices(const flexible_contain<Ts...> &) noexcept; // undefined, use in decltype

	template<class String, class T, std::size_t ...Indices>
	static constexpr void append_items(String &out, const T &t, std::index_sequence<Indices...>) {
		(..., (append(out, t.template get<Indices>()), out.push_back(',')));
	}

	template<class String, class U>
	static constexpr void append_int(String &out, bool neg, U u) {
		constexpr auto maxsz = sizeof(U) * 3 + 1; // at most 3 digits per byte + sign
		char buff[maxsz], *ptr = buff + maxsz;
		std::size_t sz = 0;
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
		if constexpr(std::is_signed_v<T>) {
			if (t == std::numeric_limits<T>::min())
				append_int(out, true, std::make_unsigned_t<T>(t));
			else
				append_int(out, (t < 0), std::make_unsigned_t<T>(t < 0 ? -t : t));
		} else {
			append_int(out, false, t);
		}
		out.push_back('}');
	}

	template<class String, std::size_t L>
	static constexpr void append(String &out, const std::integral_constant<std::size_t, L>) {
		out.append("lit<", 4);
		append_int(out, false, L);
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

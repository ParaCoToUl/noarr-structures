#ifndef NOARR_STRUCTURES_UTILITY_HPP
#define NOARR_STRUCTURES_UTILITY_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

namespace noarr {

namespace helpers {

template<class... Packs>
struct integer_sequence_concat_impl;

template<class T, T... vs1, T... vs2, class...Packs>
struct integer_sequence_concat_impl<std::integer_sequence<T, vs1...>, std::integer_sequence<T, vs2...>, Packs...> {
	using type = typename integer_sequence_concat_impl<std::integer_sequence<T, vs1..., vs2...>, Packs...>::type;
};

template<class T, T... vs1>
struct integer_sequence_concat_impl<std::integer_sequence<T, vs1...>> {
	using type = std::integer_sequence<T, vs1...>;
};

template<class Sep, class... Packs>
struct integer_sequence_concat_sep_impl;

template<class T, T... vs1, T... vs2, T... sep, class...Packs>
struct integer_sequence_concat_sep_impl<std::integer_sequence<T, sep...>, std::integer_sequence<T, vs1...>, std::integer_sequence<T, vs2...>, Packs...> {
	using type = typename integer_sequence_concat_sep_impl<std::integer_sequence<T, sep...>, std::integer_sequence<T, vs1..., vs2...>, Packs...>::type;
};

template<class T, T v1, T v2, T... vs1, T... vs2, T... sep, class...Packs>
struct integer_sequence_concat_sep_impl<std::integer_sequence<T, sep...>, std::integer_sequence<T, v1, vs1...>, std::integer_sequence<T, v2, vs2...>, Packs...> {
	using type = typename integer_sequence_concat_sep_impl<std::integer_sequence<T, sep...>, std::integer_sequence<T, v1, vs1..., sep..., v2, vs2...>, Packs...>::type;
};

template<class T, T... vs1, T... sep>
struct integer_sequence_concat_sep_impl<std::integer_sequence<T, sep...>, std::integer_sequence<T, vs1...>> {
	using type = std::integer_sequence<T, vs1...>;
};

template<class T, T V, class Pack, class = void>
struct integer_sequence_contains_impl;

template<class T, T V, T... VS>
struct integer_sequence_contains_impl<T, V, std::integer_sequence<T, V, VS...>> : std::true_type {};

template<class T, T V, T v, T... VS>
struct integer_sequence_contains_impl<T, V, std::integer_sequence<T, v, VS...>, std::enable_if_t<(V != v)>> : integer_sequence_contains_impl<T, V, std::integer_sequence<T, VS...>> {};

template<class T, T V>
struct integer_sequence_contains_impl<T, V, std::integer_sequence<T>> : std::false_type {};

} // namespace helpers

/**
 * @brief concatenates multiple integral `Packs`
 * 
 * @tparam Packs: the input integral packs
 */
template<class... Packs>
using integer_sequence_concat = typename helpers::integer_sequence_concat_impl<Packs...>::type;

/**
 * @brief concatenates multiple integral packs (the 2nd, 3rd etc. member of `Packs`) pasting the 1st member of `Packs` between each consecutive packs
 * 
 * @tparam Packs: the input integral packs, the first one is the separator used when concatenating
 */
template<class... Packs>
using integer_sequence_concat_sep = typename helpers::integer_sequence_concat_sep_impl<Packs...>::type;

/**
 * @brief an alias for std::integer_sequence<char, ...>
 * 
 * @tparam VS: the contained values
 */
template<char... VS>
using char_sequence = std::integer_sequence<char, VS...>;

template<class>
static constexpr bool always_false = false;
template<auto>
static constexpr bool value_always_false = false;

template<class T>
struct some {
	static constexpr bool present = true;
	T value;

	template<class F>
	constexpr some<decltype(std::declval<F>()(std::declval<T>()))> and_then(const F &f) const noexcept {
		return {f(value)};
	}
};

struct none {
	static constexpr bool present = false;

	template<class F>
	constexpr none and_then(const F &) const noexcept {
		return {};
	}
};

namespace constexpr_arithmetic {

template<std::size_t N>
using make_const = std::integral_constant<std::size_t, N>;

template<std::size_t A, std::size_t B>
constexpr make_const<A + B> operator+(make_const<A>, make_const<B>) noexcept { return {}; }

template<std::size_t A, std::size_t B>
constexpr make_const<A - B> operator-(make_const<A>, make_const<B>) noexcept { return {}; }

template<std::size_t A, std::size_t B>
constexpr make_const<A * B> operator*(make_const<A>, make_const<B>) noexcept { return {}; }

template<std::size_t A, std::size_t B>
constexpr make_const<A / B> operator/(make_const<A>, make_const<B>) noexcept { return {}; }

template<std::size_t A, std::size_t B>
constexpr make_const<A % B> operator%(make_const<A>, make_const<B>) noexcept { return {}; }

} // namespace constexpr_arithmetic

} // namespace noarr

#endif // NOARR_STRUCTURES_UTILITY_HPP

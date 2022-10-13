#ifndef NOARR_STRUCTURES_MANGLE_VALUE_HPP
#define NOARR_STRUCTURES_MANGLE_VALUE_HPP

#include "std_ext.hpp"

namespace noarr {

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

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_MANGLE_VALUE_HPP

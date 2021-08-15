#ifndef NOARR_STRUCTURES_MANGLE_VALUE_HPP
#define NOARR_STRUCTURES_MANGLE_VALUE_HPP

#include "std_ext.hpp"

namespace noarr {

namespace helpers {

template<typename T, T V, typename = void>
struct mangle_value_impl;

template<typename T, T V>
using mangle_value = typename mangle_value_impl<T, V>::type;

template<typename T, T V, typename Acc = integral_pack<char>, typename = void>
struct mangle_integral_impl;

template<typename T, T V>
using mangle_integral = typename mangle_integral_impl<T, V>::type;

template<typename T, char... Acc, T V>
struct mangle_integral_impl<T, V, char_pack<Acc...>, std::enable_if_t<(V >= 10)>> {
	using type = typename mangle_integral_impl<T, V / 10, char_pack<(char)(V % 10) + '0', Acc...>>::type;
};

template<typename T, char... Acc, T V>
struct mangle_integral_impl<T, V, char_pack<Acc...>, std::enable_if_t<(V < 10 && V >= 0)>> {
	using type = char_pack<(char)(V % 10) + '0', Acc...>;
};

template<typename T, T V>
struct mangle_value_impl<T, V, std::enable_if_t<std::is_integral<T>::value>> {
	using type = mangle_integral<T, V>;
};

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_MANGLE_VALUE_HPP

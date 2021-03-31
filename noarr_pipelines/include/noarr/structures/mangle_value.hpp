#ifndef NOARR_MANGLE_VALUE_HPP
#define NOARR_MANGLE_VALUE_HPP

#include "noarr_std_ext.hpp"

namespace noarr {

template<typename T, T V, typename = void>
struct _mangle_value;

template<typename T, T V>
using mangle_value = typename _mangle_value<T, V>::type;

template<typename T, T V, typename Acc = integral_pack<char>, typename = void>
struct _mangle_integral;

template<typename T, T V>
using mangle_integral = typename _mangle_integral<T, V>::type;

template<typename T, char... Acc, T V>
struct _mangle_integral<T, V, char_pack<Acc...>, std::enable_if_t<(V >= 10)>> {
    using type = typename _mangle_integral<T, V / 10, char_pack<(char)(V % 10) + '0', Acc...>>::type;
};

template<typename T, char... Acc, T V>
struct _mangle_integral<T, V, char_pack<Acc...>, std::enable_if_t<(V < 10 && V >= 0)>> {
    using type = char_pack<(char)(V % 10) + '0', Acc...>;
};

template<typename T, T V>
struct _mangle_value<T, V, std::enable_if_t<std::is_integral<T>::value>> {
    using type = mangle_integral<T, V>;
};

}

#endif // NOARR_MANGLE_VALUE_HPP
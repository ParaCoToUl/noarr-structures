#ifndef NOARR_STRUCTURES_SCALAR_NAMES_HPP
#define NOARR_STRUCTURES_SCALAR_NAMES_HPP

#include "std_ext.hpp"
#include "mangle_value.hpp"

namespace noarr {

template<typename T, typename = void>
struct scalar_name;

template<typename T>
using scalar_name_t = typename scalar_name<T>::type;

template<typename T, typename>
struct scalar_name {
    static_assert(template_false<T>::value, "scalar_name<T> has to be implemented");
    using type = void;
};

template<typename T>
struct scalar_name<T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>> {
    using type = integral_pack_concat<char_pack<'i'>, mangle_value<int, 8 * sizeof(T)>>;
};

template<typename T>
struct scalar_name<T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>> {
    using type = integral_pack_concat<char_pack<'u'>, mangle_value<int, 8 * sizeof(T)>>;
};

template<typename T>
struct scalar_name<T, std::enable_if_t<std::is_floating_point<T>::value>> {
    using type = integral_pack_concat<char_pack<'f'>, mangle_value<int, 8 * sizeof(T)>>;
};

}

#endif // NOARR_STRUCTURES_SCALAR_NAMES_HPP
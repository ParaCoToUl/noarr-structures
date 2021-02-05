#ifndef NOARR_SCALAR_NAMES_HPP
#define NOARR_SCALAR_NAMES_HPP

#include "noarr_std_ext.hpp"

namespace noarr {

template<typename T, typename = void>
struct scalar_name;

template<typename T>
using scalar_name_t = typename scalar_name<T>::type;

template<typename T, typename>
struct scalar_name {
    static_assert(template_false<T>::value, "scalar_name<T> has to be implemented");
};

template<>
struct scalar_name<double> {
    using type = integral_pack<char, 'd', 'o', 'u', 'b', 'l', 'e'>;
};

template<>
struct scalar_name<float> {
    using type = integral_pack<char, 'f', 'l', 'o', 'a', 't'>;
};

template<>
struct scalar_name<int> {
    using type = integral_pack<char, 'i', 'n', 't'>;
};

}

#endif // NOARR_SCALAR_NAMES_HPP
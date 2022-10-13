#ifndef NOARR_STRUCTURES_SCALAR_NAMES_HPP
#define NOARR_STRUCTURES_SCALAR_NAMES_HPP

#include "std_ext.hpp"
#include "mangle_value.hpp"

namespace noarr {

namespace helpers {

/**
 * @brief returns a textual representation of a scalar type using `char_sequence`
 * 
 * @tparam T: the scalar type
 */
template<class T, class = void>
struct scalar_name;

template<class T>
using scalar_name_t = typename scalar_name<T>::type;

template<class T, class>
struct scalar_name {
	static_assert(always_false<T>, "scalar_name<T> has to be implemented");
	using type = void;
};

template<class T>
struct scalar_name<T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>> {
	using type = integer_sequence_concat<char_sequence<'i'>, mangle_value<int, 8 * sizeof(T)>>;
};

template<class T>
struct scalar_name<T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>> {
	using type = integer_sequence_concat<char_sequence<'u'>, mangle_value<int, 8 * sizeof(T)>>;
};

template<class T>
struct scalar_name<T, std::enable_if_t<std::is_floating_point<T>::value>> {
	using type = integer_sequence_concat<char_sequence<'f'>, mangle_value<int, 8 * sizeof(T)>>;
};

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_SCALAR_NAMES_HPP

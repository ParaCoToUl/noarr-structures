#ifndef NOARR_STRUCTURES_STD_EXT_HPP
#define NOARR_STRUCTURES_STD_EXT_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

namespace noarr {

/**
 * @brief returns the first of the given types
 * 
 * @tparam T: the type to be returned
 * @tparam TS: the types to be forgotten
 */
template<class T, class... TS>
using first_t = T;

template<class R, class T>
constexpr decltype(auto) const_v(R &&ret, const T &&) noexcept {
	return std::forward<R>(ret);
}

/**
 * @brief a shortcut for applying std::remove_cv and std::remove_reference
 * 
 * @tparam T: the original type
 */
template<class T>
using remove_cvref = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

/**
 * @brief The field `::value` contains whether the given type is a C-array
 * 
 * @tparam T: the type
 */
template<class T>
struct is_array : std::false_type {};

template<class T, std::size_t N>
struct is_array<T[N]> : std::true_type {};

/**
 * @brief Contains multiple values of a certain type
 * 
 * @tparam T: the type of the values
 * @tparam VS: the values
 */
template<class T, T... VS>
struct integral_pack;

namespace helpers {

template<class T, T V, class Pack, class = void>
struct integral_pack_contains_impl;

}

/**
 * @brief returns whether an integral pack contains a certain value
 * 
 * @tparam Pack: the hay pack
 * @tparam V: the needle value
 */
template<class Pack, typename Pack::value_type V>
using integral_pack_contains = helpers::integral_pack_contains_impl<typename Pack::value_type, V, Pack>;

/**
 * @brief contains a set of scalar values `VS` of type `T`
 * 
 * @tparam T: the scalar type of the values
 * @tparam VS: the set of the contained values
 */
template<class T, T... VS>
struct integral_pack {
	using value_type = T;
	using type = integral_pack;

	template<T V>
	struct contains : integral_pack_contains<integral_pack, V> {};
};

namespace helpers {

template<class... Packs>
struct integral_pack_concat_impl;

template<class T, T... vs1, T... vs2, class...Packs>
struct integral_pack_concat_impl<integral_pack<T, vs1...>, integral_pack<T, vs2...>, Packs...> {
	using type = typename integral_pack_concat_impl<integral_pack<T, vs1..., vs2...>, Packs...>::type;
};

template<class T, T... vs1>
struct integral_pack_concat_impl<integral_pack<T, vs1...>> {
	using type = integral_pack<T, vs1...>;
};

template<class Sep, class... Packs>
struct integral_pack_concat_sep_impl;

template<class T, T... vs1, T... vs2, T... sep, class...Packs>
struct integral_pack_concat_sep_impl<integral_pack<T, sep...>, integral_pack<T, vs1...>, integral_pack<T, vs2...>, Packs...> {
	using type = typename integral_pack_concat_sep_impl<integral_pack<T, sep...>, integral_pack<T, vs1..., vs2...>, Packs...>::type;
};

template<class T, T v1, T v2, T... vs1, T... vs2, T... sep, class...Packs>
struct integral_pack_concat_sep_impl<integral_pack<T, sep...>, integral_pack<T, v1, vs1...>, integral_pack<T, v2, vs2...>, Packs...> {
	using type = typename integral_pack_concat_sep_impl<integral_pack<T, sep...>, integral_pack<T, v1, vs1..., sep..., v2, vs2...>, Packs...>::type;
};

template<class T, T... vs1, T... sep>
struct integral_pack_concat_sep_impl<integral_pack<T, sep...>, integral_pack<T, vs1...>> {
	using type = integral_pack<T, vs1...>;
};

}

/**
 * @brief concatenates multiple integral `Packs`
 * 
 * @tparam Packs: the input integral packs
 */
template<class... Packs>
using integral_pack_concat = typename helpers::integral_pack_concat_impl<Packs...>::type;

/**
 * @brief concatenates multiple integral packs (the 2nd, 3rd etc. member of `Packs`) pasting the 1st member of `Packs` between each consecutive packs
 * 
 * @tparam Packs: the input integral packs, the first one is the separator used when concatenating
 */
template<class... Packs>
using integral_pack_concat_sep = typename helpers::integral_pack_concat_sep_impl<Packs...>::type;

namespace helpers {

template<class T, T V, T... VS>
struct integral_pack_contains_impl<T, V, integral_pack<T, V, VS...>> : std::true_type {};

template<class T, T V, T v, T... VS>
struct integral_pack_contains_impl<T, V, integral_pack<T, v, VS...>, std::enable_if_t<(V != v)>> : integral_pack_contains_impl<T, V, integral_pack<T, VS...>> {};

template<class T, T V>
struct integral_pack_contains_impl<T, V, integral_pack<T>> : std::false_type {};

}

/**
 * @brief an alias for integral_pack<char, ...>
 * 
 * @tparam VS: the contained values
 */
template<char... VS>
using char_pack = integral_pack<char, VS...>;

/**
 * @brief used for lazily retrieving a false `::value`
 * 
 * @tparam T: the type preceding the value
 */
template<class T>
struct template_false : std::false_type {};

namespace helpers {

template<template<class> class Function, class Tuple, class = void>
struct tuple_forall_impl : std::false_type {};

template<template<class> class Function, class T, class... TS>
struct tuple_forall_impl<Function, std::tuple<T, TS...>, std::enable_if_t<Function<T>::value>> : tuple_forall_impl<Function, std::tuple<TS...>> {};

template<template<class> class Function>
struct tuple_forall_impl<Function, std::tuple<>> : std::true_type {};

}

/**
 * @brief checks whether a type function applied to all elements of a tuple contains always true `::value`
 * 
 * @tparam Function: the applied type function
 * @tparam Tuple: the tuple containing the set of inputs for the function
 */
template<template<class> class Function, class Tuple>
using tuple_forall = helpers::tuple_forall_impl<Function, Tuple>;

template<class>
static constexpr bool always_false = false;
template<char>
static constexpr bool always_false_dim = false;

} // namespace noarr

#endif // NOARR_STRUCTURES_STD_EXT_HPP

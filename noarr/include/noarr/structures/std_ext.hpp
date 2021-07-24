#ifndef NOARR_STRUCTURES_STD_EXT_HPP
#define NOARR_STRUCTURES_STD_EXT_HPP

#include <cstddef>
#include <type_traits>

namespace noarr {

template<class... T>
using void_t = void;

template<class T>
using remove_cvref = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

/**
 * @brief The field `::value` contains whether the given type is a C-array
 * 
 * @tparam T: the type
 */
template<class T>
struct is_array {
    using value_type = bool;
    static constexpr value_type value = false;
};

template<class T, std::size_t N>
struct is_array<T[N]> {
    using value_type = bool;
    static constexpr value_type value = true;
};

/**
 * @brief Contains multiple values of a certain type
 * 
 * @tparam T: the type of the values
 * @tparam VS: the values
 */
template<class T, T... VS>
struct integral_pack;

template<class T, T V, class Pack, typename = void>
struct _integral_pack_contains;

/**
 * @brief returns whether an integral pack contains a certain value
 * 
 * @tparam Pack: the hay pack
 * @tparam V: the needle value
 */
template<class Pack, typename Pack::value_type V>
using integral_pack_contains = _integral_pack_contains<typename Pack::value_type, V, Pack>;

template<class T, T... VS>
struct integral_pack {
    using value_type = T;
    using type = integral_pack;

    template<T V>
    struct contains {
        static constexpr bool value = integral_pack_contains<integral_pack, V>::value;
    };
};

template<class... Packs>
struct _integral_pack_concat;

template<class T, T... vs1, T... vs2, class...Packs>
struct _integral_pack_concat<integral_pack<T, vs1...>, integral_pack<T, vs2...>, Packs...> {
    using type = typename _integral_pack_concat<integral_pack<T, vs1..., vs2...>, Packs...>::type;
};

template<class T, T... vs1>
struct _integral_pack_concat<integral_pack<T, vs1...>> {
    using type = integral_pack<T, vs1...>;
};

template<class Sep, class... Packs>
struct _integral_pack_concat_sep;

template<class T, T... vs1, T... vs2, T... sep, class...Packs>
struct _integral_pack_concat_sep<integral_pack<T, sep...>, integral_pack<T, vs1...>, integral_pack<T, vs2...>, Packs...> {
    using type = typename _integral_pack_concat_sep<integral_pack<T, sep...>, integral_pack<T, vs1..., vs2...>, Packs...>::type;
};

template<class T, T v1, T v2, T... vs1, T... vs2, T... sep, class...Packs>
struct _integral_pack_concat_sep<integral_pack<T, sep...>, integral_pack<T, v1, vs1...>, integral_pack<T, v2, vs2...>, Packs...> {
    using type = typename _integral_pack_concat_sep<integral_pack<T, sep...>, integral_pack<T, v1, vs1..., sep..., v2, vs2...>, Packs...>::type;
};

template<class T, T... vs1, T... sep>
struct _integral_pack_concat_sep<integral_pack<T, sep...>, integral_pack<T, vs1...>> {
    using type = integral_pack<T, vs1...>;
};

/**
 * @brief concatenates multiple integral packs
 * 
 * @tparam Packs 
 */
template<class... Packs>
using integral_pack_concat = typename _integral_pack_concat<Packs...>::type;

template<class... Packs>
using integral_pack_concat_sep = typename _integral_pack_concat_sep<Packs...>::type;

template<class T, T V, T... VS>
struct _integral_pack_contains<T, V, integral_pack<T, V, VS...>> {
    static constexpr bool value = true;
};

template<class T, T V, T v, T... VS>
struct _integral_pack_contains<T, V, integral_pack<T, v, VS...>, std::enable_if_t<(V != v)>> {
    static constexpr bool value = _integral_pack_contains<T, V, integral_pack<T, VS...>>::value;
};

template<class T, T V>
struct _integral_pack_contains<T, V, integral_pack<T>> {
    static constexpr bool value = false;
};

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
template<typename T>
struct template_false {
    static constexpr bool value = false;
};

template<template<typename> class Function, typename Tuple, typename = void>
struct _tuple_forall {
    static constexpr bool value = false;
};

template<template<typename> class Function, typename T, typename... TS>
struct _tuple_forall<Function, std::tuple<T, TS...>, std::enable_if_t<Function<T>::value>> {
    static constexpr bool value = _tuple_forall<Function, std::tuple<TS...>>::value;
};

template<template<typename> class Function>
struct _tuple_forall<Function, std::tuple<>> {
    static constexpr bool value = true;
};

/**
 * @brief checks whether a type function applied to all elements of a tuple contains always true `::value`
 * 
 * @tparam Function: the applied type function
 * @tparam Tuple: the tuple containing the set of inputs for the function
 */
template<template<typename> class Function, typename Tuple>
using tuple_forall = _tuple_forall<Function, Tuple>;

} // namespace noarr

#endif // NOARR_STRUCTURES_STD_EXT_HPP

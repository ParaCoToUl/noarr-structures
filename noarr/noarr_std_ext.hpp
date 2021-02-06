#ifndef NOARR_STD_EXT_HPP
#define NOARR_STD_EXT_HPP

#include <cstddef>
#include <type_traits>

namespace noarr {

template<class... T>
using void_t = void;

template<class T>
using remove_cvref = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

// TODO: this is not really cosher
struct empty_struct_t {
    constexpr empty_struct_t() : _value{} {}
    const char _value[0];
};

template <typename T>
using is_empty = typename std::is_base_of<empty_struct_t, T>;

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

template<class T, T... vs>
struct integral_pack;

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

template<class... Packs>
using integral_pack_concat = typename _integral_pack_concat<Packs...>::type;

template<class... Packs>
using integral_pack_concat_sep = typename _integral_pack_concat_sep<Packs...>::type;

template<typename T>
struct template_false {
    static constexpr bool value = false;
};

} // namespace noarr

#endif // NOARR_STD_EXT_HPP

#ifndef NOARR_IO_HPP
#define NOARR_IO_HPP

#include <ostream>

#include "noarr_core.hpp"
#include "noarr_std_ext.hpp"
#include "noarr_structs.hpp"
#include "noarr_scalar_names.hpp"

namespace noarr {

template<typename T, typename = void>
struct struct_name;

template<typename T, typename Pre = integral_pack<char>, typename Post = integral_pack<char>>
struct _mangle;

template<typename T>
using struct_name_t = typename struct_name<T>::type;

template<typename T>
using mangle = typename _mangle<T>::type;

template<typename T, typename>
struct struct_name {
    static_assert(template_false<T>::value, "the name of struct T has to be implemented");
};

// TODO: check if integral_pack
template<typename T>
struct struct_name<T, void_t<typename T::name>> {
    using type = typename T::name;
};

template<typename T>
struct scalar_name<T, void_t<mangle<T>>> {
    using type = mangle<T>;
};

template<typename T, typename Pre, typename Post>
struct _mangle {
    using type = integral_pack_concat<Pre, struct_name_t<T>, integral_pack<char, '<'>, mangle<typename sub_structures<T>::value_type>, integral_pack<char, '>'>, Post>;
};

template<typename T1, typename T2, typename... Ts, typename Pre, typename Post>
struct _mangle<std::tuple<T1, T2, Ts...>, Pre, Post> {
    using type = integral_pack_concat<Pre, mangle<T1>, integral_pack<char, ','>, mangle<std::tuple<T2, Ts...>>, Post>;
};

template<typename T, typename Pre, typename Post>
struct _mangle<scalar<T>, Pre, Post> {
    using type = integral_pack_concat<Pre, struct_name_t<scalar<T>>, integral_pack<char, '<'>, scalar_name_t<T>, integral_pack<char, '>'>, Post>;
};

template<typename T, typename Pre, typename Post>
struct _mangle<std::tuple<T>, Pre, Post> {
    using type = typename _mangle<T, Pre, Post>::type;
};

template<typename Pre, typename Post>
struct _mangle<std::tuple<>, Pre, Post> {
    using type = integral_pack_concat<Pre, Post>;
};

template<typename T>
struct _print_struct;

template<char... Name>
struct _print_struct<integral_pack<char, Name...>> {
    static constexpr std::ostream &print(std::ostream &out) {
        constexpr char name[] = {Name..., '\0'};
        return out << name;
    }
};

template<typename T>
inline constexpr std::ostream &print_struct(std::ostream &out, T) {
    return _print_struct<mangle<T>>::print(out);
}



}

#endif // NOARR_IO_HPP
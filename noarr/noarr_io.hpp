#ifndef NOARR_IO_HPP
#define NOARR_IO_HPP

#include <ostream>

#include "noarr_core.hpp"
#include "noarr_std_ext.hpp"
#include "noarr_structs.hpp"
#include "noarr_mangle_value.hpp"
#include "noarr_scalar_names.hpp"
#include "noarr_struct_desc.hpp"

namespace noarr {

template<typename T, typename = void>
struct get_struct_desc;

template<typename T, typename Pre = integral_pack<char>, typename Post = integral_pack<char>>
struct _mangle;

template<typename T>
using get_struct_desc_t = typename get_struct_desc<T>::type;

template<typename T>
using mangle = typename _mangle<T>::type;

template<typename T, typename>
struct get_struct_desc {
    static_assert(template_false<T>::value, "the T::desc (or get_struct_desc_<T>) has to be implemented");
};

// TODO: check if integral_pack
template<typename T>
struct get_struct_desc<T, void_t<typename T::desc>> {
    using type = typename T::desc;
};

template<typename T>
struct scalar_name<T, void_t<mangle<T>>> {
    using type = mangle<T>;
};

template<typename T, typename Pre, typename Post>
struct _mangle {
    using type = typename _mangle<get_struct_desc_t<T>, Pre, Post>::type;
};

template<typename T, typename Pre, typename Post>
struct _mangle_scalar;

template<typename Dims, typename Acc = integral_pack<char>>
struct _transform_dims;

template<typename... Dims>
using transform_dims = typename _transform_dims<Dims...>::type;

template<char Dim1, char Dim2, char... Dims, char... Acc>
struct _transform_dims<dims_impl<Dim1, Dim2, Dims...>, integral_pack<char, Acc...>> {
    using type = typename _transform_dims<dims_impl<Dim2, Dims...>, integral_pack<char, '\'', Dim1, '\'', ',', Acc...>>::type;
};

template<char Dim, char... Acc>
struct _transform_dims<dims_impl<Dim>, integral_pack<char, Acc...>> {
    using type = integral_pack<char, '\'', Dim, '\'', Acc...>;
};

template<typename Acc>
struct _transform_dims<dims_impl<>, Acc> {
    using type = Acc;
};

template<typename Name, typename Dims, typename ADims, typename... Params, typename Pre, typename Post>
struct _mangle<struct_desc<Name, Dims, ADims, Params...>, Pre, Post> {
    using type = integral_pack_concat<
        Pre,
        Name,
        integral_pack<char, '<'>,
        integral_pack_concat_sep<integral_pack<char, ','>, transform_dims<Dims>, transform_dims<ADims>, mangle<Params>...>,
        integral_pack<char, '>'>,
        Post>;
};

template<typename Name, typename Param, typename Pre, typename Post>
struct _mangle_scalar<struct_desc<Name, dims_impl<>, dims_impl<>, type_param<Param>>, Pre, Post> {
    using type = integral_pack_concat<Pre, Name, integral_pack<char, '<'>, scalar_name_t<Param>, integral_pack<char, '>'>, Post>;
};

template<typename T, typename Pre, typename Post>
struct _mangle<type_param<T>, Pre, Post> {
    using type = integral_pack_concat<Pre, mangle<T>, Post>;
};

template<typename T, T V, typename Pre, typename Post>
struct _mangle<value_param<T, V>, Pre, Post> {
    using type = integral_pack_concat<Pre, mangle_value<T, V>, Post>;
};

template<typename T, typename Pre, typename Post>
struct _mangle<scalar<T>, Pre, Post> {
    using type = typename _mangle_scalar<get_struct_desc_t<scalar<T>>, Pre, Post>::type;
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
#ifndef NOARR_STRUCTURES_MANGLE_HPP
#define NOARR_STRUCTURES_MANGLE_HPP

#include "std_ext.hpp"
#include "scalar.hpp"
#include "mangle_value.hpp"
#include "scalar_names.hpp"
#include "struct_decls.hpp"

namespace noarr {

template<typename T, typename Pre = char_pack<>, typename Post = char_pack<>>
struct _mangle;

template<typename T>
using mangle = typename _mangle<T>::type;

template<typename T>
struct scalar_name<T, void_t<get_struct_desc_t<T>>> {
    using type = mangle<T>;
};

template<typename T, typename Pre, typename Post>
struct _mangle {
    using type = typename _mangle<get_struct_desc_t<T>, Pre, Post>::type;
};

template<typename T, typename Pre, typename Post>
struct _mangle_scalar;

template<typename Dims>
struct _transform_dims;

template<typename Dims>
using transform_dims = typename _transform_dims<Dims>::type;

template<char Dim>
struct _transform_dims<dims_impl<Dim>> {
    using type = char_pack<'\'', Dim, '\''>;
};

template<>
struct _transform_dims<dims_impl<>> {
    using type = dims_impl<>;
};

template<typename Name, typename Dims, typename ADims, typename... Params, typename Pre, typename Post>
struct _mangle<struct_description<Name, Dims, ADims, Params...>, Pre, Post> {
    using type = integral_pack_concat<
        Pre,
        Name,
        char_pack<'<'>,
        integral_pack_concat_sep<char_pack<','>, transform_dims<Dims>, transform_dims<ADims>, mangle<Params>...>,
        char_pack<'>'>,
        Post>;
};

template<typename Name, typename Param, typename Pre, typename Post>
struct _mangle_scalar<struct_description<Name, dims_impl<>, dims_impl<>, type_param<Param>>, Pre, Post> {
    using type = integral_pack_concat<Pre, Name, char_pack<'<'>, scalar_name_t<Param>, char_pack<'>'>, Post>;
};

template<typename T, typename Pre, typename Post>
struct _mangle<type_param<T>, Pre, Post> {
    using type = integral_pack_concat<Pre, mangle<T>, Post>;
};

template<typename T, T V, typename Pre, typename Post>
struct _mangle<value_param<T, V>, Pre, Post> {
    using type = integral_pack_concat<Pre, char_pack<'('>, scalar_name_t<T>, char_pack<')'>, mangle_value<T, V>, Post>;
};

template<typename T, typename Pre, typename Post>
struct _mangle<scalar<T>, Pre, Post> {
    using type = typename _mangle_scalar<get_struct_desc_t<scalar<T>>, Pre, Post>::type;
};

template<typename Pre, typename Post>
struct _mangle<std::tuple<>, Pre, Post> {
    using type = integral_pack_concat<Pre, Post>;
};

}

#endif // NOARR_STRUCTURES_MANGLE_HPP

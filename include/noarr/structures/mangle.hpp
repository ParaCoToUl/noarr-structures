#ifndef NOARR_STRUCTURES_MANGLE_HPP
#define NOARR_STRUCTURES_MANGLE_HPP

#include "std_ext.hpp"
#include "scalar.hpp"
#include "mangle_value.hpp"
#include "scalar_names.hpp"
#include "struct_decls.hpp"

namespace noarr {

namespace helpers {

template<class T, class Pre = char_pack<>, class Post = char_pack<>>
struct mangle_impl;

template<class Dims>
struct transform_dims_impl;

}

/**
 * @brief Returns a textual representation of the type of a structure using `char_pack`
 * 
 * @tparam T: the structure
 */
template<class T>
using mangle = typename helpers::mangle_impl<T>::type;

/**
 * @brief encloses dimensions in single quotes
 * 
 * @tparam Dims: the dimensions to be transformed
 */
template<class Dims>
using transform_dims = typename helpers::transform_dims_impl<Dims>::type;

namespace helpers {

template<class T>
struct scalar_name<T, void_t<get_struct_desc_t<T>>> {
	using type = mangle<T>;
};

template<class T, class Pre, class Post>
struct mangle_impl : mangle_impl<get_struct_desc_t<T>, Pre, Post> {};

template<class T, class Pre, class Post>
struct mangle_scalar;

template<char Dim>
struct transform_dims_impl<dims_impl<Dim>> {
	using type = char_pack<'\'', Dim, '\''>;
};

template<>
struct transform_dims_impl<dims_impl<>> {
	using type = dims_impl<>;
};

template<class Name, class Dims, class ADims, class... Params, class Pre, class Post>
struct mangle_impl<struct_description<Name, Dims, ADims, Params...>, Pre, Post>
	: integral_pack_concat<
		Pre,
		Name,
		char_pack<'<'>,
		integral_pack_concat_sep<char_pack<','>, transform_dims<Dims>, transform_dims<ADims>, mangle<Params>...>,
		char_pack<'>'>,
		Post> {};

template<class Name, class Param, class Pre, class Post>
struct mangle_scalar<struct_description<Name, dims_impl<>, dims_impl<>, type_param<Param>>, Pre, Post>
	: integral_pack_concat<Pre, Name, char_pack<'<'>, scalar_name_t<Param>, char_pack<'>'>, Post> {};

template<class T, class Pre, class Post>
struct mangle_impl<structure_param<T>, Pre, Post>
	: integral_pack_concat<Pre, mangle<T>, Post> {};

template<class T, class Pre, class Post>
struct mangle_impl<type_param<T>, Pre, Post>
	: integral_pack_concat<Pre, scalar_name_t<T>, Post> {};

template<class T, T V, class Pre, class Post>
struct mangle_impl<value_param<T, V>, Pre, Post>
	: integral_pack_concat<Pre, char_pack<'('>, scalar_name_t<T>, char_pack<')'>, mangle_value<T, V>, Post> {};

template<class T, class Pre, class Post>
struct mangle_impl<scalar<T>, Pre, Post>
	: mangle_scalar<get_struct_desc_t<scalar<T>>, Pre, Post> {};

template<class Pre, class Post>
struct mangle_impl<std::tuple<>, Pre, Post>
	: integral_pack_concat<Pre, Post> {};

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_MANGLE_HPP

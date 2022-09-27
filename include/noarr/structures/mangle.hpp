#ifndef NOARR_STRUCTURES_MANGLE_HPP
#define NOARR_STRUCTURES_MANGLE_HPP

#include "std_ext.hpp"
#include "scalar.hpp"
#include "mangle_value.hpp"
#include "scalar_names.hpp"
#include "struct_decls.hpp"

namespace noarr {

namespace helpers {

template<class T>
struct mangle_impl;

}

/**
 * @brief Returns a textual representation of the type of a structure using `char_pack`
 * 
 * @tparam T: the structure
 */
template<class T>
using mangle = typename helpers::mangle_impl<T>::type;

namespace helpers {

template<class T>
struct scalar_name<T, std::void_t<get_struct_desc_t<T>>> {
	using type = mangle<T>;
};

template<class T>
struct mangle_impl : mangle_impl<get_struct_desc_t<T>> {};

template<class Name, class... Params>
struct mangle_impl<struct_description<Name, Params...>>
	: integral_pack_concat<
		Name,
		char_pack<'<'>,
		integral_pack_concat_sep<char_pack<','>, mangle<Params>...>,
		char_pack<'>'>> {};

template<class T>
struct mangle_impl<structure_param<T>>
	: integral_pack_concat<mangle<T>> {};

template<class T>
struct mangle_impl<type_param<T>>
	: integral_pack_concat<scalar_name_t<T>> {};

template<class T, T V>
struct mangle_impl<value_param<T, V>>
	: integral_pack_concat<char_pack<'('>, scalar_name_t<T>, char_pack<')'>, mangle_value<T, V>> {};

template<char Dim>
struct mangle_impl<dim_param<Dim>>
	: integral_pack_concat<char_pack<'\'', Dim, '\''>> {};

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_MANGLE_HPP

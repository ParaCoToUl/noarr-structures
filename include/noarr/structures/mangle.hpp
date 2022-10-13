#ifndef NOARR_STRUCTURES_MANGLE_HPP
#define NOARR_STRUCTURES_MANGLE_HPP

#include "std_ext.hpp"
#include "scalar.hpp"
#include "mangle_value.hpp"
#include "scalar_names.hpp"
#include "struct_decls.hpp"

namespace noarr {

namespace helpers {

template<const char Name[], class Indices, class Params>
struct mangle_desc;

}

/**
 * @brief Returns a textual representation of the type of a structure using `char_sequence`
 * 
 * @tparam T: the structure
 */
template<class T>
using mangle = typename helpers::mangle_desc<T::name, std::make_index_sequence<sizeof(T::name) - 1>, typename T::params>::type;

namespace helpers {

template<class T>
struct mangle_param;

template<class T>
struct mangle_param<structure_param<T>> { using type = integer_sequence_concat<mangle<T>>; };

template<class T>
struct mangle_param<type_param<T>> { using type = integer_sequence_concat<scalar_name_t<T>>; };

template<class T, T V>
struct mangle_param<value_param<T, V>> { using type = integer_sequence_concat<char_sequence<'('>, scalar_name_t<T>, char_sequence<')'>, mangle_value<T, V>>; };

template<char Dim>
struct mangle_param<dim_param<Dim>> { using type = integer_sequence_concat<char_sequence<'\'', Dim, '\''>>; };

template<const char Name[], std::size_t... Indices, class... Params>
struct mangle_desc<Name, std::index_sequence<Indices...>, struct_params<Params...>> {
	using type = integer_sequence_concat<
		char_sequence<Name[Indices]..., '<'>,
		integer_sequence_concat_sep<char_sequence<','>, typename mangle_param<Params>::type...>,
		char_sequence<'>'>>;
};

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_MANGLE_HPP

#ifndef NOARR_STRUCTURES_STRUCT_DECLS_HPP
#define NOARR_STRUCTURES_STRUCT_DECLS_HPP

#include <tuple>

#include "std_ext.hpp"

namespace noarr {

/**
 * @brief a struct that describes a structure
 * 
 * @tparam Name: the name of the structure
 * @tparam Dims: the dimensions the structure introduces
 * @tparam ADims: the dimensions the structure consumes
 * @tparam Params: template parameters of the structure
 */
template<class Name, class Dims, class ADims, class... Params>
struct struct_description {
	using name = Name;
	using dims = Dims;
	using adims = ADims;
	using description = struct_description;
};

template<class>
struct structure_param;

template<class>
struct type_param;

template<class T, T V>
struct value_param;

/**
 * @brief returns the `struct_description` of a structure
 * 
 * @tparam T: the structure
 * @tparam class: a placeholder type
 */
template<class T, class = void>
struct get_struct_desc;

template<class T>
using get_struct_desc_t = typename get_struct_desc<T>::type;

template<class T>
struct get_struct_desc<T, std::void_t<typename T::description>> {
	using type = typename T::description;
};

/**
 * @brief The type that holds all the dimensions of a structure
 * 
 * @tparam Dims: the dimensions
 */
template<char... Dims>
using dims_impl = char_pack<Dims...>;

/**
 * @brief returns the dimensions introduced by the structure
 * 
 * @tparam T: the structure
 */
template<class T>
using get_dims = typename T::description::dims;

template<class StructInner, class StructOuter, class State>
constexpr std::size_t offset_of(StructOuter structure, State state) noexcept {
	if constexpr(std::is_same_v<StructInner, StructOuter>) {
		// TODO check that state only contains relevant lengths
		return 0;
	} else {
		return structure.template strict_offset_of<StructInner>(state);
	}
}

template<class StructInner, class StructOuter, class State>
constexpr auto state_at(StructOuter structure, State state) noexcept {
	if constexpr(std::is_same_v<StructInner, StructOuter>) {
		return state;
	} else {
		return structure.template strict_state_at<StructInner>(state);
	}
}

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_DECLS_HPP

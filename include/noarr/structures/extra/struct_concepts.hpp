#ifndef NOARR_STRUCTURES_STRUCT_CONCEPTS_HPP
#define NOARR_STRUCTURES_STRUCT_CONCEPTS_HPP

#include "../base/structs_common.hpp"
#include "../extra/to_struct.hpp"

namespace noarr {

namespace helpers {

template<class T, IsDim auto... Dims>
struct has_dims {
	using struct_type = typename to_struct<T>::type;
	using dim_tree = sig_dim_tree<typename struct_type::signature>;
	using restricted = dim_tree_restrict<dim_tree, dim_sequence<Dims...>>;

	static constexpr bool value = (... && dim_tree_contains<Dims, dim_tree>);
};

} // namespace helpers

template<class T, IsDim auto... Dims>
concept HasDims = requires(T) {
	typename to_struct<T>::type;
} && helpers::has_dims<T, Dims...>::value;

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_CONCEPTS_HPP

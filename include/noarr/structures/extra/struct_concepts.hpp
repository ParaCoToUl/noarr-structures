#ifndef NOARR_STRUCTURES_STRUCT_CONCEPTS_HPP
#define NOARR_STRUCTURES_STRUCT_CONCEPTS_HPP

#include "../base/utility.hpp"
#include "../extra/sig_utils.hpp"
#include "../extra/to_struct.hpp"

namespace noarr {

namespace helpers {

template<class T, auto... Dims>
requires IsDimPack<decltype(Dims)...>
struct has_dims {
private:
	using struct_type = typename to_struct<T>::type;
	using dim_tree_helper = sig_dim_tree<typename struct_type::signature>;

public:
	static constexpr bool value = (... && dim_tree_contains<Dims, dim_tree_helper>);
};

} // namespace helpers

template<class T, auto... Dims>
concept HasDims = IsDimPack<decltype(Dims)...> && requires(T) { typename to_struct<T>::type; } &&
                  helpers::has_dims<T, Dims...>::value;

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCT_CONCEPTS_HPP

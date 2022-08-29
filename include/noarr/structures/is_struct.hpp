#ifndef NOARR_STRUCTURES_IS_STRUCT_HPP
#define NOARR_STRUCTURES_IS_STRUCT_HPP

#include "signature.hpp"

namespace noarr {

namespace helpers {

template<class T, class = void>
struct is_struct_impl : std::false_type {};
template<class T>
struct is_struct_impl<T, std::void_t<typename T::signature>> : std::true_type {
	static_assert(is_signature<typename T::signature>());
};

} // namespace helpers

/**
 * @brief returns whether the type `T` meets the criteria for structures
 * 
 * @tparam T: the input type
 */
template<class T>
using is_struct = helpers::is_struct_impl<T>;

} // namespace noarr

#endif // NOARR_STRUCTURES_IS_STRUCT_HPP

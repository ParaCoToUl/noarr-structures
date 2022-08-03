#ifndef NOARR_STRUCTURES_IS_STRUCT_HPP
#define NOARR_STRUCTURES_IS_STRUCT_HPP

#include "std_ext.hpp"
#include "pipes.hpp"

namespace noarr {

namespace helpers {

template<class T, class = void>
struct is_struct_impl : std::false_type {};

} // namespace helpers

/**
 * @brief returns whether the type `T` meets the criteria for structures
 * 
 * @tparam T: the input type
 */
template<class T>
using is_struct = helpers::is_struct_impl<T>;

namespace helpers {

template<class T>
struct is_struct_impl<T, void_t<decltype(std::declval<T>().sub_structures())>> : std::true_type {};

} // namespace helpers

} // namespace noarr

#endif // NOARR_STRUCTURES_IS_STRUCT_HPP

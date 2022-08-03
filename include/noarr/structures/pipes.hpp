#ifndef NOARR_STRUCTURES_PIPES_HPP
#define NOARR_STRUCTURES_PIPES_HPP

#include "std_ext.hpp"
#include "struct_decls.hpp"
#include "is_struct.hpp"

namespace noarr {

/**
 * @brief performs a simple application of `F` to `S`
 * 
 * @tparam S: the structure type
 * @tparam F: the function type
 * @param s: the structure
 * @param f: the function
 * @return the result of the piping
 */
template<class S, class F>
constexpr auto operator|(S s, F f) noexcept ->
std::enable_if_t<is_struct<std::enable_if_t<std::is_class<S>::value, S>>::value, decltype(std::declval<F>()(std::declval<S>()))> {
	return f(s);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_PIPES_HPP

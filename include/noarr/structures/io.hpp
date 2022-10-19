#ifndef NOARR_STRUCTURES_IO_HPP
#define NOARR_STRUCTURES_IO_HPP

#include <ostream>

#include "utility.hpp"
#include "mangle.hpp"

namespace noarr {

namespace helpers {

template<class T>
struct print_struct_impl;

template<char... Name>
struct print_struct_impl<char_sequence<Name...>> {
	// translates a `char_sequence<Name...>` to the corresponding c string
	static constexpr char name[] = {Name..., '\0'};
	static constexpr std::ostream &print(std::ostream &out) noexcept {
		return out << name;
	}
};

} // namespace helpers

/**
 * @brief outputs the textual representation of the structure's type to the given `std::ostream`
 * 
 * @tparam T: the input strucure
 * @param out: the output stream
 */
template<class T>
constexpr std::ostream &print_struct(std::ostream &out, T) noexcept {
	return helpers::print_struct_impl<mangle<T>>::print(out);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_IO_HPP

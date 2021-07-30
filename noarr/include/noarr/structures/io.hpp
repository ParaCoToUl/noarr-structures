#ifndef NOARR_STRUCTURES_IO_HPP
#define NOARR_STRUCTURES_IO_HPP

#include <ostream>

#include "std_ext.hpp"
#include "mangle.hpp"

namespace noarr {

namespace helpers {

template<typename T>
struct print_struct_impl;

template<char... Name>
struct print_struct_impl<char_pack<Name...>> {
    static constexpr std::ostream &print(std::ostream &out) {
        constexpr const char name[] = {Name..., '\0'};
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
template<typename T>
inline constexpr std::ostream &print_struct(std::ostream &out, T) {
    return print_struct_impl<mangle<T>>::print(out);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_IO_HPP

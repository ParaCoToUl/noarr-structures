#ifndef NOARR_STRUCTURES_IO_HPP
#define NOARR_STRUCTURES_IO_HPP

#include <ostream>

#include "std_ext.hpp"
#include "mangle.hpp"

namespace noarr {

template<typename T>
struct _print_struct;

template<char... Name>
struct _print_struct<char_pack<Name...>> {
    static constexpr std::ostream &print(std::ostream &out) {
        constexpr const char name[] = {Name..., '\0'};
        return out << name;
    }
};

/**
 * @brief outputs the textual representation of the structure's type to the given `std::ostream`
 * 
 * @tparam T: the input strucure
 * @param out: the output stream 
 */
template<typename T>
inline constexpr std::ostream &print_struct(std::ostream &out, T) {
    return _print_struct<mangle<T>>::print(out);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_IO_HPP

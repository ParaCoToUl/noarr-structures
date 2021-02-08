#ifndef NOARR_IO_HPP
#define NOARR_IO_HPP

#include <ostream>

#include "noarr_std_ext.hpp"
#include "noarr_mangle.hpp"

namespace noarr {

template<typename T>
struct _print_struct;

template<char... Name>
struct _print_struct<char_pack<Name...>> {
    static constexpr std::ostream &print(std::ostream &out) {
        constexpr char name[] = {Name..., '\0'};
        return out << name;
    }
};

template<typename T>
inline constexpr std::ostream &print_struct(std::ostream &out, T) {
    return _print_struct<mangle<T>>::print(out);
}

}

#endif // NOARR_IO_HPP

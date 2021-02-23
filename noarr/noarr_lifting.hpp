#ifndef NOARR_LIFTING_HPP
#define NOARR_LIFTING_HPP

#include <istream>
#include <ostream>
#include <type_traits>

#include "noarr_std_ext.hpp"
#include "noarr_is_struct.hpp"
#include "noarr_struct_traits.hpp"

namespace noarr {

template<typename Struct>
inline constexpr auto lift(std::istream &in, Struct s, char *offset) -> std::enable_if_t<is_dynamic_dimension<Struct>::value, std::istream&>{
    for (std::size_t i = 0; i < s.length(); ++i) {
        lift(in, std::get<0>(s.sub_structures()), offset + s.offset(i));
    }

    return in;
}

template<std::size_t I, typename Struct>
struct lift_static {
    static constexpr std::istream &lift_(std::istream &in, Struct s, char *offset) {
        lift_static<I - 1, Struct>::lift_(in, s, offset);
        return lift(in, std::get<I - 1>(s.sub_structures()), offset + s.template offset<I>());
    }
};

template<typename Struct>
struct lift_static<0, Struct> {
    static constexpr std::istream &lift_(std::istream &in, Struct, char*) {
        return in;
    }
};

template<typename Struct>
inline constexpr auto lift(std::istream &in, Struct s, char *offset) -> std::enable_if_t<!is_dynamic_dimension<Struct>::value && is_static_dimension<Struct>::value, std::istream&>{
    return lift_static<s.length(), Struct>::lift_(in, s, offset);
}

template<typename Struct>
inline constexpr auto lift(std::istream &in, Struct s, char *offset) -> std::enable_if_t<is_point<Struct>::value, std::istream&>{
    return lift(in, std::get<0>(s.sub_structures()), offset + s.offset());
}

template<typename Type>
inline constexpr std::istream &lift(std::istream &in, scalar<Type>, char *offset) {
    return in >> *reinterpret_cast<Type*>(offset);
}

template<typename Struct>
inline constexpr auto unlift(std::ostream &out, Struct s, char *offset) -> std::enable_if_t<is_dynamic_dimension<Struct>::value, std::ostream&>{
    for (std::size_t i = 0; i < s.length(); ++i) {
        unlift(out, std::get<0>(s.sub_structures()), offset + s.offset(i));
    }

    return out;
}

template<std::size_t I, typename Struct>
struct unlift_static {
    static constexpr std::ostream &unlift_(std::ostream &out, Struct s, char *offset) {
        unlift_static<I - 1, Struct>::unlift_(out, s, offset);
        return unlift(out, std::get<I - 1>(s.sub_structures()), offset + s.template offset<I>());
    }
};

template<typename Struct>
struct unlift_static<0, Struct> {
    static constexpr std::ostream &unlift_(std::ostream &out, Struct, char*) {
        return out;
    }
};

template<typename Struct>
inline constexpr auto unlift(std::ostream &out, Struct s, char *offset) -> std::enable_if_t<!is_dynamic_dimension<Struct>::value && is_static_dimension<Struct>::value, std::ostream&>{
    return unlift_static<s.length(), Struct>::unlift_(out, s, offset);
}

template<typename Struct>
inline constexpr auto unlift(std::ostream &out, Struct s, char *offset) -> std::enable_if_t<is_point<Struct>::value, std::ostream&>{
    return unlift(out, std::get<0>(s.sub_structures()), offset + s.offset());
}

template<typename Type>
inline constexpr std::ostream &unlift(std::ostream &out, scalar<Type>, char *offset) {
    return out << *reinterpret_cast<Type*>(offset);
}

}

#endif // NOARR_LIFTING_HPP
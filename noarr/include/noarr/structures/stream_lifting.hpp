#ifndef NOARR_STRUCTURES_STREAM_LIFTING_HPP
#define NOARR_STRUCTURES_STREAM_LIFTING_HPP

#include <istream>
#include <ostream>
#include <type_traits>

#include "std_ext.hpp"
#include "is_struct.hpp"
#include "struct_traits.hpp"

namespace noarr {

template<typename Struct>
inline constexpr auto stream_lift(std::istream &in, Struct s, char *offset) -> std::enable_if_t<is_dynamic_dimension<Struct>::value, std::istream&>{
    for (std::size_t i = 0; i < s.length(); ++i) {
        stream_lift(in, std::get<0>(s.sub_structures()), offset + s.offset(i));
    }

    return in;
}

template<std::size_t I, typename Struct>
struct stream_lift_static {
    static constexpr std::istream &stream_lift_(std::istream &in, Struct s, char *offset) {
        stream_lift_static<I - 1, Struct>::stream_lift_(in, s, offset);
        return stream_lift(in, std::get<I - 1>(s.sub_structures()), offset + s.template offset<I>());
    }
};

template<typename Struct>
struct stream_lift_static<0, Struct> {
    static constexpr std::istream &stream_lift_(std::istream &in, Struct, char*) {
        return in;
    }
};

template<typename Struct>
inline constexpr auto stream_lift(std::istream &in, Struct s, char *offset) -> std::enable_if_t<!is_dynamic_dimension<Struct>::value && is_static_dimension<Struct>::value, std::istream&>{
    return stream_lift_static<s.length(), Struct>::stream_lift_(in, s, offset);
}

template<typename Struct>
inline constexpr auto stream_lift(std::istream &in, Struct s, char *offset) -> std::enable_if_t<is_point<Struct>::value, std::istream&>{
    return stream_lift(in, std::get<0>(s.sub_structures()), offset + s.offset());
}

template<typename Type>
inline constexpr std::istream &stream_lift(std::istream &in, scalar<Type>, char *offset) {
    return in >> *reinterpret_cast<Type*>(offset);
}

template<typename Struct>
inline constexpr auto stream_unlift(std::ostream &out, Struct s, char *offset) -> std::enable_if_t<is_dynamic_dimension<Struct>::value, std::ostream&>{
    for (std::size_t i = 0; i < s.length(); ++i) {
        stream_unlift(out, std::get<0>(s.sub_structures()), offset + s.offset(i));
    }

    return out;
}

template<std::size_t I, typename Struct>
struct stream_unlift_static {
    static constexpr std::ostream &stream_unlift_(std::ostream &out, Struct s, char *offset) {
        stream_unlift_static<I - 1, Struct>::stream_unlift_(out, s, offset);
        return stream_unlift(out, std::get<I - 1>(s.sub_structures()), offset + s.template offset<I>());
    }
};

template<typename Struct>
struct stream_unlift_static<0, Struct> {
    static constexpr std::ostream &stream_unlift_(std::ostream &out, Struct, char*) {
        return out;
    }
};

template<typename Struct>
inline constexpr auto stream_unlift(std::ostream &out, Struct s, char *offset) -> std::enable_if_t<!is_dynamic_dimension<Struct>::value && is_static_dimension<Struct>::value, std::ostream&>{
    return stream_unlift_static<s.length(), Struct>::stream_unlift_(out, s, offset);
}

template<typename Struct>
inline constexpr auto stream_unlift(std::ostream &out, Struct s, char *offset) -> std::enable_if_t<is_point<Struct>::value, std::ostream&>{
    return stream_unlift(out, std::get<0>(s.sub_structures()), offset + s.offset());
}

template<typename Type>
inline constexpr std::ostream &stream_unlift(std::ostream &out, scalar<Type>, char *offset) {
    return out << *reinterpret_cast<Type*>(offset);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_STREAM_LIFTING_HPP
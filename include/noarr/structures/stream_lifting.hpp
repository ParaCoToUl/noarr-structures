#ifndef NOARR_STRUCTURES_STREAM_LIFTING_HPP
#define NOARR_STRUCTURES_STREAM_LIFTING_HPP

#include <istream>
#include <ostream>
#include <type_traits>

#include "std_ext.hpp"
#include "is_struct.hpp"
#include "struct_traits.hpp"

namespace noarr {

template<class Struct>
constexpr auto stream_lift(std::istream &in, Struct s, char *offset) noexcept -> std::enable_if_t<is_dynamic_dimension<Struct>::value, std::istream&> {
	for (std::size_t i = 0; i < s.length(); ++i) {
		stream_lift(in, std::get<0>(s.sub_structures()), offset + s.offset(i));
	}

	return in;
}

namespace helpers {

template<std::size_t I, class Struct>
struct stream_lift_static {
	static constexpr std::istream &stream_lift(std::istream &in, Struct s, char *offset) noexcept {
		stream_lift_static<I - 1, Struct>::stream_lift(in, s, offset);
		return stream_lift(in, std::get<I - 1>(s.sub_structures()), offset + s.template offset<I>());
	}
};

template<class Struct>
struct stream_lift_static<0, Struct> {
	static constexpr std::istream &stream_lift(std::istream &in, Struct, char *) noexcept {
		return in;
	}
};

} // namespace helpers

template<class Struct>
constexpr auto stream_lift(std::istream &in, Struct s, char *offset) noexcept -> std::enable_if_t<!is_dynamic_dimension<Struct>::value && is_static_dimension<Struct>::value, std::istream&> {
	return helpers::stream_lift_static<s.length(), Struct>::stream_lift(in, s, offset);
}

template<class Struct>
constexpr auto stream_lift(std::istream &in, Struct s, char *offset) noexcept -> std::enable_if_t<is_point<Struct>::value, std::istream&> {
	return stream_lift(in, std::get<0>(s.sub_structures()), offset + s.offset());
}

template<class Type>
constexpr std::istream &stream_lift(std::istream &in, scalar<Type>, char *offset) noexcept {
	return in >> *reinterpret_cast<Type*>(offset);
}

template<class Struct>
constexpr auto stream_unlift(std::ostream &out, Struct s, const char *offset) noexcept -> std::enable_if_t<is_dynamic_dimension<Struct>::value, std::ostream&> {
	for (std::size_t i = 0; i < s.length(); ++i) {
		stream_unlift(out, std::get<0>(s.sub_structures()), offset + s.offset(i));
	}

	return out;
}

namespace helpers {

template<std::size_t I, class Struct>
struct stream_unlift_static {
	static constexpr std::ostream &stream_unlift(std::ostream &out, Struct s, const char *offset) noexcept {
		stream_unlift_static<I - 1, Struct>::stream_unlift(out, s, offset);
		return stream_unlift(out, std::get<I - 1>(s.sub_structures()), offset + s.template offset<I>());
	}
};

template<class Struct>
struct stream_unlift_static<0, Struct> {
	static constexpr std::ostream &stream_unlift(std::ostream &out, Struct, const char *) noexcept {
		return out;
	}
};

} // namespace helpers

template<class Struct>
constexpr auto stream_unlift(std::ostream &out, Struct s, const char *offset) noexcept -> std::enable_if_t<!is_dynamic_dimension<Struct>::value && is_static_dimension<Struct>::value, std::ostream&> {
	return helpers::stream_unlift_static<s.length(), Struct>::stream_unlift(out, s, offset);
}

template<class Struct>
constexpr auto stream_unlift(std::ostream &out, Struct s, const char *offset) noexcept -> std::enable_if_t<is_point<Struct>::value, std::ostream&> {
	return stream_unlift(out, std::get<0>(s.sub_structures()), offset + s.offset());
}

template<class Type>
constexpr std::ostream &stream_unlift(std::ostream &out, scalar<Type>, const char *offset) noexcept {
	return out << *reinterpret_cast<const Type*>(offset);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_STREAM_LIFTING_HPP

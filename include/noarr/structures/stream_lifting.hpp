#ifndef NOARR_STRUCTURES_STREAM_LIFTING_HPP
#define NOARR_STRUCTURES_STREAM_LIFTING_HPP

#include <istream>
#include <ostream>

#include "traverser.hpp"

namespace noarr {

template<class Struct>
constexpr std::istream &stream_lift(std::istream &in, Struct s, void *data) noexcept {
	traverser(s).for_each([&in, s, data](auto state) noexcept {
		in >> (s | get_at(data, state));
	});
	return in;
}

template<class Struct>
constexpr std::ostream &stream_unlift(std::ostream &out, Struct s, const void *data) noexcept {
	traverser(s).for_each([&out, s, data](auto state) noexcept {
		out << (s | get_at(data, state)) << '\n';
	});
	return out;
}

} // namespace noarr

#endif // NOARR_STRUCTURES_STREAM_LIFTING_HPP

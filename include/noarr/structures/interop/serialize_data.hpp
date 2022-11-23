#ifndef NOARR_STRUCTURES_SERIALIZE_DATA_HPP
#define NOARR_STRUCTURES_SERIALIZE_DATA_HPP

#include <istream>
#include <ostream>

#include "../extra/traverser.hpp"

namespace noarr {

template<class Struct>
constexpr std::istream &deserialize_data(std::istream &in, Struct s, void *data) noexcept {
	traverser(s).for_each([&in, s, data](auto state) noexcept {
		in >> (s | get_at(data, state));
	});
	return in;
}

template<class Struct>
constexpr std::ostream &serialize_data(std::ostream &out, Struct s, const void *data) noexcept {
	traverser(s).for_each([&out, s, data](auto state) noexcept {
		out << (s | get_at(data, state)) << '\n';
	});
	return out;
}

} // namespace noarr

#endif // NOARR_STRUCTURES_SERIALIZE_DATA_HPP

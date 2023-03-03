#ifndef NOARR_STRUCTURES_SERIALIZE_DATA_HPP
#define NOARR_STRUCTURES_SERIALIZE_DATA_HPP

#include <istream>
#include <ostream>

#include "../extra/traverser.hpp"
#include "../interop/bag.hpp"

namespace noarr {

template<class Struct>
constexpr std::istream &deserialize_data(std::istream &in, Struct s, void *data) noexcept {
	traverser(s).for_each([&in, s, data](auto state) noexcept {
		in >> (s | get_at(data, state));
	});
	return in;
}

template<class Bag>
constexpr std::istream &deserialize_data(std::istream &in, Bag &&bag) noexcept {
	return deserialize_data(in, bag.structure().unwrap(), bag.data());
}

template<class Struct>
constexpr std::ostream &serialize_data(std::ostream &out, Struct s, const void *data) noexcept {
	traverser(s).for_each([&out, s, data](auto state) noexcept {
		out << (s | get_at(data, state)) << '\n';
	});
	return out;
}

template<class Bag>
constexpr std::ostream &serialize_data(std::ostream &out, const Bag &bag) noexcept {
	return serialize_data(out, bag.structure().unwrap(), bag.data());
}

} // namespace noarr

#endif // NOARR_STRUCTURES_SERIALIZE_DATA_HPP

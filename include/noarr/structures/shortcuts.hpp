#ifndef NOARR_STRUCTURES_SHORTCUTS_HPP
#define NOARR_STRUCTURES_SHORTCUTS_HPP

#include "blocks.hpp"
#include "reorder.hpp"

namespace noarr {

template<char Dim, char DimMajor, char DimMinor, class... OptionalMinorLengthT>
constexpr auto strip_mine(OptionalMinorLengthT... optional_minor_length) {
	return into_blocks<Dim, DimMajor, DimMinor>(optional_minor_length...) ^ hoist<DimMajor>();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_SHORTCUTS_HPP

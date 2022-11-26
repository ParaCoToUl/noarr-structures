#ifndef NOARR_STRUCTURES_TBB_HPP
#define NOARR_STRUCTURES_TBB_HPP

#include <tbb/tbb.h>

#include "../interop/traverser_iter.hpp"

namespace noarr {

// declared in traverser_iter.hpp
template<char Dim, class Struct, class Order>
template<class Split>
constexpr traverser_range_t<Dim, Struct, Order>::traverser_range_t(traverser_range_t &orig, Split) noexcept : base(orig), begin_idx(orig.begin_idx + (orig.end_idx - orig.begin_idx) / 2), end_idx(orig.end_idx) {
	static_assert(std::is_same_v<Split, tbb::split>, "Invalid constructor call");
	orig.end_idx = begin_idx;
}

} // namespace noarr

#endif // NOARR_STRUCTURES_TBB_HPP
